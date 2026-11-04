"""
Periodic 3D crystal plasticity RVE for FE2.

Fluctuation formulation: u = (F_bar - I) X + w, with w periodic. One evaluation:

  1. Total gradient per cell: H = H_bar + grad(w).
  2. P, its tangent A = dP/dH and the updated history from cp_model.py (JAX).
  3. Newton on w until the periodic residual vanishes.
  4. Homogenized stress: volume average of P.
  5. Homogenized tangent dP_bar/dF_bar, by the implicit function theorem: with K the
     converged Jacobian, one solve per component of H_bar on the same factorization,

         K dw_kl = -G^T D E_kl,    A_bar[:, kl] = < A : (E_kl + grad(dw_kl)) >

The mesh is linear tets with one integration point, so everything is written with
one sparse operator G, the gradient per cell in the 9 components of H (row-major),
acting on the reduced dofs:

    R = G^T (V P),   K = G^T D G,   D = block-diagonal V A

The reduced dofs are the periodic root nodes, minus corner 0 (at the origin), which is
fixed to remove the rigid translation. Each slave surface then follows its master
surface plus the corner displacement given by H_bar. Which H_bar components are
prescribed, and so how rigid rotation is removed, is decided by the caller.
"""

import os

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc

import cp_model
from cp_model import CPParams, update_and_tangent_batched

_HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY_9 = np.eye(3).ravel()


def read_tess_quaternions(tess_file):
    """Grain orientations from the *ori section of a Neper .tess file."""
    with open(tess_file) as f:
        lines = f.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "*ori")
    assert "quaternion" in lines[start + 1], "expected quaternion descriptor"
    quats = []
    for line in lines[start + 2:]:
        if line.lstrip().startswith("*"):
            break
        quats.append([float(v) for v in line.split()])
    return np.array(quats)


def read_msh_periodicity(msh_file):
    """0-based periodic (slave, master) node pairs from Neper's $Periodicity section."""
    with open(msh_file) as f:
        lines = f.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "$Periodicity")
    pairs = [line.split()[:2] for line in lines[start + 2 : start + 2 + int(lines[start + 1])]]
    return np.array(pairs, dtype=np.int64) - 1


def read_msh_mesh(msh_file):
    """Points, TET4 connectivity, and 0-based grain index per cell."""
    import meshio
    mesh = meshio.read(msh_file)
    blocks = [(b.data, phys) for b, phys in zip(mesh.cells, mesh.cell_data["gmsh:physical"]) if b.type == "tetra"]
    cells = np.vstack([b[0] for b in blocks])
    grains = np.concatenate([b[1] for b in blocks]).astype(np.int32) - 1
    return np.array(mesh.points), cells, grains


def periodic_roots(n_nodes, pairs):
    """Root node of each node, with union-find so that chains cannot be missed."""
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for slave, master in pairs:
        parent[find(slave)] = find(master)
    return np.array([find(i) for i in range(n_nodes)])


def _lu_solver(comm=MPI.COMM_SELF):
    ksp = PETSc.KSP().create(comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    return ksp


class Micromodel:
    def __init__(self, points=None, cells=None, grains=None, quats=None, pairs=None,
                 msh_file=None, tess_file=None,
                 slip_sys_file=os.path.join(_HERE, "input_slip_sys.txt"),
                 newton_rtol=1e-8, newton_atol=1e-7, newton_max_it=30, max_line_search=6,
                 chunk_size=1024, verbose=False):
        if points is None:
            msh_file = msh_file or os.path.join(_HERE, "rve", "rve100.msh")
            tess_file = tess_file or os.path.join(_HERE, "rve", "rve100.tess")
            points, cells, grains = read_msh_mesh(msh_file)
            quats = read_tess_quaternions(tess_file)
            pairs = read_msh_periodicity(msh_file)

        self.verbose = verbose
        self.newton_rtol = newton_rtol
        self.newton_atol = newton_atol
        self.newton_max_it = newton_max_it
        self.max_line_search = max_line_search
        self.chunk_size = chunk_size

        self.points, self.cells = points, cells
        self.n_cells = cells.shape[0]
        self.dN, self.volumes = self._shape_gradients(points, cells)
        self.vol = self.volumes.sum()

        self._build_gradient_operator(points, cells, pairs)
        self._setup_material(grains, quats, slip_sys_file)

        self.A_ksp = _lu_solver()
        self.mat = None
        self.rhs, self.sol = None, None

    # --------------------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------------------

    @staticmethod
    def _shape_gradients(points, cells):
        """Constant shape function gradients (cell, node, dim) and cell volumes."""
        edges = points[cells[:, 1:]] - points[cells[:, [0]]]
        volumes = np.abs(np.linalg.det(edges)) / 6.0
        dN_123 = np.linalg.inv(edges).transpose(0, 2, 1)  # (cell, node 1-3, dim)
        dN_0 = -dN_123.sum(axis=1, keepdims=True)
        return np.concatenate([dN_0, dN_123], axis=1), volumes

    def _build_gradient_operator(self, points, cells, pairs):
        n_nodes = points.shape[0]
        roots = periodic_roots(n_nodes, pairs)

        # Corner 0 is the fixed node. All corners are periodic images of each other, so
        # they share one root, and fixing corner 0 gives w = 0 at all of them.
        corner_0 = np.argmin(np.linalg.norm(points, axis=1))
        kept = np.setdiff1d(np.unique(roots), roots[corner_0])
        node_to_red = -np.ones(n_nodes, dtype=np.int64)
        node_to_red[kept] = np.arange(kept.size)
        node_to_red = node_to_red[roots]
        self.n_dofs = 3 * kept.size
        self.node_to_red = node_to_red

        # G[9 c + 3 i + j, 3 red(a) + i] = dN_a,j
        rows, cols, vals = [], [], []
        for a in range(4):
            red = node_to_red[cells[:, a]]
            for i in range(3):
                for j in range(3):
                    keep = red >= 0
                    rows.append(9 * np.arange(self.n_cells)[keep] + 3 * i + j)
                    cols.append(3 * red[keep] + i)
                    vals.append(self.dN[keep, a, j])
        self.G = sp.csr_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(9 * self.n_cells, self.n_dofs))
        self.G.sum_duplicates()
        self.G_T = self.G.T.tocsr()

        # Sparsity of the block-diagonal D
        base = 9 * np.arange(self.n_cells)
        self._D_rows = (base[:, None, None] + np.arange(9)[None, :, None] * np.ones((1, 1, 9), dtype=np.int64)).ravel()
        self._D_cols = (base[:, None, None] + np.arange(9)[None, None, :] * np.ones((1, 9, 1), dtype=np.int64)).ravel()

    def _setup_material(self, grains, quats, slip_sys_file):
        schmid = cp_model.schmid_tensors(slip_sys_file)
        n_slip = schmid.shape[0]
        self.params = CPParams(q=np.ones((n_slip, n_slip)))   # latent hardening r = 1
        C_rot, schmid_rot = cp_model.rotated_tensors(cp_model.rotation_matrices(quats), cp_model.cubic_stiffness(), schmid)
        self.C_cells = C_rot[grains]
        self.schmid_cells = schmid_rot[grains]
        self.n_slip = n_slip

    def initial_history(self):
        return {"Fp_inv": np.tile(np.eye(3), (self.n_cells, 1, 1)),
                "g": np.full((self.n_cells, self.n_slip), cp_model.GSS_INITIAL),
                "slip": np.zeros((self.n_cells, self.n_slip)),
                "S": np.zeros((self.n_cells, 3, 3))}

    # --------------------------------------------------------------------
    # Micro problem
    # --------------------------------------------------------------------

    def _update(self, w):
        """Constitutive update at fluctuation w. Returns (ok, residual vector)."""
        H = self.H_bar + (self.G @ w).reshape(self.n_cells, 3, 3)

        ok = self._constitutive(H, self.S_guess)
        if not ok and self.S_guess.any():
            ok = self._constitutive(H, np.zeros_like(self.S_guess))   # safe restart from S = 0
        if ok:
            self.S_guess = self.new_history["S"]
        return ok, self.G_T @ (self.volumes[:, None] * self.P.reshape(self.n_cells, 9)).ravel()

    def _constitutive(self, H, S_guess):
        """P, A and trial history for all cells. Returns False if a point failed."""
        hist = self.history

        # Chunks: each one stops at its own slowest point, and large batches can
        # stall the multi-threaded JAX CPU runtime
        chunks = []
        for start in range(0, self.n_cells, self.chunk_size):
            c = slice(start, start + self.chunk_size)
            chunks.append(update_and_tangent_batched(
                H[c], S_guess[c], hist["Fp_inv"][c], hist["g"][c], hist["slip"][c],
                self.C_cells[c], self.schmid_cells[c], self.dt, self.params))
        joined = lambda arrays: np.concatenate([np.asarray(a) for a in arrays])

        self.P = joined([ch[1][0] for ch in chunks])
        self.A = joined([ch[0] for ch in chunks]).reshape(self.n_cells, 9, 9)
        self.new_history = {"Fp_inv": joined([ch[1][1] for ch in chunks]),
                            "g": joined([ch[1][2] for ch in chunks]),
                            "slip": joined([ch[1][3] for ch in chunks]),
                            "S": joined([ch[1][5] for ch in chunks])}
        return bool(joined([ch[1][4] for ch in chunks]).all()) and bool(np.isfinite(self.P).all())

    def _material_block(self):
        data = (self.volumes[:, None, None] * self.A).ravel()
        return sp.csr_array((data, (self._D_rows, self._D_cols)), shape=(self.G.shape[0], self.G.shape[0]))

    def _factorize(self):
        """Assembles K = G^T D G and hands it to the LU solver."""
        self.D = self._material_block()
        K = (self.G_T @ self.D @ self.G).tocsr()
        K.sort_indices()
        if self.mat is None:
            self.mat = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr.astype(PETSc.IntType), K.indices.astype(PETSc.IntType), K.data), comm=MPI.COMM_SELF)
            self.rhs, self.sol = self.mat.createVecRight(), self.mat.createVecRight()
            self._K_pattern = (K.indptr.copy(), K.indices.copy())
        else:
            self.mat.zeroEntries()
            self.mat.setValuesCSR(K.indptr.astype(PETSc.IntType), K.indices.astype(PETSc.IntType), K.data)
        self.mat.assemble()
        self.A_ksp.setOperators(self.mat)

    def _solve(self, b):
        self.rhs.setArray(b)
        self.A_ksp.solve(self.rhs, self.sol)
        return self.sol.getArray(readonly=True).copy()

    def _solve_micro_newton(self, w):
        ok, R = self._update(w)
        if not ok:
            return False, w
        res = res0 = np.linalg.norm(R)

        for it in range(self.newton_max_it):
            if self.verbose:
                print(f"[Micro] [Iter {it:2d}] residual {res:.4e}", flush=True)
            if res <= self.newton_atol or res <= self.newton_rtol * res0:
                return True, w

            self._factorize()
            dw = self._solve(-R)

            # Backtrack if the full step fails or increases the residual
            alpha = 1.0
            for _ in range(self.max_line_search):
                ok, R_new = self._update(w + alpha * dw)
                res_new = np.linalg.norm(R_new)
                if ok and res_new < res:
                    break
                alpha *= 0.5
            else:
                return False, w
            w, R, res = w + alpha * dw, R_new, res_new

        return False, w

    # --------------------------------------------------------------------
    # Macro-scale coupling
    # --------------------------------------------------------------------

    def evaluate_homogenized_properties(self, H_bar, history, dt, w_init=None):
        """
        Solves the RVE for a macro displacement gradient H_bar (3x3) and a committed
        history. Returns (converged, P_bar (3x3), A_bar (9x9), w, new_history).
        After convergence, self.dw_dH holds the sensitivities dw/dH_bar, which give a
        first-order prediction of w for a nearby H_bar.
        """
        self.H_bar = np.asarray(H_bar, dtype=np.float64).reshape(1, 3, 3)
        self.history = history
        self.S_guess = history["S"]
        self.dt = float(dt)
        w0 = np.zeros(self.n_dofs) if w_init is None else w_init

        converged, w = self._solve_micro_newton(w0)
        if not converged:
            return False, np.zeros((3, 3)), np.zeros((9, 9)), w0, history

        return True, self.compute_stress(), self.compute_tangent(), w, self.new_history

    def total_displacement(self, H_bar, w):
        """Nodal displacement (H_bar X + w), for visualization."""
        w_nodes = np.zeros_like(self.points)
        free = self.node_to_red >= 0
        w_nodes[free] = w.reshape(-1, 3)[self.node_to_red[free]]
        return self.points @ np.asarray(H_bar).T + w_nodes

    def compute_stress(self):
        return (self.volumes[:, None, None] * self.P).sum(axis=0) / self.vol

    def compute_tangent(self):
        """Exact dP_bar/dH_bar (see the module docstring)."""
        self._factorize()
        V_A = self.volumes[:, None, None] * self.A

        A_bar = V_A.sum(axis=0)
        self.dw_dH = np.zeros((self.n_dofs, 9))
        for kl in range(9):
            f = self.G_T @ V_A[:, :, kl].ravel()
            dw = self._solve(-f)
            self.dw_dH[:, kl] = dw
            A_bar[:, kl] += np.einsum("cij,cj->i", V_A, (self.G @ dw).reshape(self.n_cells, 9))
        return A_bar / self.vol