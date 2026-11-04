# Esi: 304 steel CPFEM on a 100-grain periodic RVE with periodic boundary
# conditions (PBC) and GENERAL mixed macroscopic control.
#
# Same constitutive model, material parameters, and dt as the original
# example `CPFEM/polycrystal_304steel/` (models_304steel.py is a verbatim
# copy; it is NOT modified).
#
# Architecture (PBC and loading control are fully separated):
#
#   * PBC (this file, fixed): fluctuation formulation
#         u(X) = (F_bar - I) X + w(X),
#     w periodic on Neper's periodic node pairs (anti-periodic tractions
#     follow from the reduced weak form), imposed with the prolongation
#     matrix P_mat of JAX-FEM's applications/periodic_bc example. The
#     macroscopic gradient H_bar = F_bar - I enters the (unmodified)
#     material model as an extra internal variable, so the constitutive
#     update sees the total gradient H_bar + grad(w). Rigid translation is
#     removed by pinning w at one interior unpaired node.
#
#   * Macroscopic loading (macro_loading.py, swappable): a history is a
#     list of MacroStep objects -- per increment, any subset of the nine
#     H_bar components is prescribed (arbitrary time histories: monotonic,
#     cyclic, nonproportional, random walk, ...), and the remaining
#     components (from the symmetric slot set) are unknowns determined so
#     that selected volume-averaged Cauchy stress components hit target
#     values (mixed control).
#
#   * Outer iteration (run_history below): multivariate quasi-Newton
#     (Broyden) on the unknown H_bar components, seeded with the exact
#     elastic Jacobian assembled from the model's stiffness tensor C.
#     PATH DEPENDENCE IS RESPECTED: every trial FE solve within one
#     increment restarts from the same committed internal state
#     (Fp_inv, slip resistance, accumulated slip, rotations); internal
#     variables are committed ONLY after both the FE residual and the
#     macroscopic stress-control residual have converged.
#
# Presets (see macro_loading.PRESETS):
#   uniaxial-strain   F_bar fully prescribed, diag(1,1,1+eps(t)) --
#                     validation mode (hydrostatic sigma_zz grows unbounded
#                     because isochoric crystal slip cannot relax it)
#   uniaxial-stress   H_zz prescribed; H_xx, H_yy INDEPENDENT unknowns and
#                     the three symmetric shear slots unknowns, targets
#                     <sigma_xx>=<sigma_yy>=<sigma_xy>=<sigma_xz>=<sigma_yz>=0
#   cyclic            triangular eps_zz(t) under uniaxial-stress control
#   nonproportional   tension ramp, then held tension + prescribed shear
#   random-walk       fully prescribed random symmetric H_bar increments
#
# Run (env `cpfem-esi`):
#   python problem.py --preset uniaxial-stress            # full schedule
#   python problem.py --preset uniaxial-strain --steps 2  # smoke test
#
# Outputs per run (results/):
#   macro_<name>.csv        per increment: full F_bar (9), volume-averaged
#                           Cauchy tensor (6), mean stress, von Mises of the
#                           averaged tensor, average local von Mises,
#                           macro-iteration count and final residual
#   macro_<name>_iters.csv  convergence history of the macro iteration
#   u_final_<name>.vtu      final-step fields

import argparse
import os
import sys
import time

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as np
import numpy as onp
import scipy.sparse
import meshio

from jax import config
config.update("jax_enable_x64", True)

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.utils import save_sol

crt_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, crt_dir)
from models_304steel import CrystalPlasticity   # verbatim copy, unmodified
from macro_loading import PRESETS

rve_dir = os.path.join(crt_dir, 'rve')
results_dir = os.path.join(crt_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

MACRO_TOL = 1.0          # |R|_inf tolerance on stress targets [MPa]
MACRO_MAX_ITERS = 10     # FE solves per increment budget for the macro loop
MACRO_MAX_HALVINGS = 2   # step-halving guard inside the macro Newton update

# ---------------------------------------------------------------------------
# Neper file parsing
# ---------------------------------------------------------------------------

def read_tess_quaternions(tess_file):
    """Grain orientations from the *ori section of a Neper .tess file
    (scalar-first quaternions, one line per grain)."""
    quats = []
    with open(tess_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == '*ori':
            assert 'quaternion' in lines[i + 1], "expected quaternion descriptor"
            j = i + 2
            while j < len(lines) and not lines[j].lstrip().startswith('*'):
                quats.append([float(v) for v in lines[j].split()])
                j += 1
            break
    quats = onp.array(quats)
    assert quats.shape[1] == 4, f"bad quaternion array shape {quats.shape}"
    return quats


def read_msh_periodicity(msh_file):
    """Periodic node pairs from the $Periodicity section Neper writes into
    its .msh file. Each line: slave master shift_x shift_y shift_z
    (1-based node ids, integer lattice shifts). Returns 0-based
    (slaves, masters, shifts)."""
    slaves, masters, shifts = [], [], []
    with open(msh_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == '$Periodicity':
            n_pairs = int(lines[i + 1])
            for k in range(n_pairs):
                vals = lines[i + 2 + k].split()
                slaves.append(int(vals[0]) - 1)
                masters.append(int(vals[1]) - 1)
                shifts.append([int(v) for v in vals[2:5]])
            break
    else:
        raise ValueError(f"No $Periodicity section in {msh_file}")
    return onp.array(slaves), onp.array(masters), onp.array(shifts)


def read_msh_mesh(msh_file):
    """Points, TET4 connectivity, and 0-based grain index per cell."""
    meshio_mesh = meshio.read(msh_file)
    cells_list, gids_list = [], []
    for block, phys in zip(meshio_mesh.cells, meshio_mesh.cell_data['gmsh:physical']):
        if block.type == 'tetra':
            cells_list.append(block.data)
            gids_list.append(phys)
    cells = onp.vstack(cells_list)
    cell_grain_inds = onp.concatenate(gids_list).astype(onp.int32) - 1
    return onp.array(meshio_mesh.points), cells, cell_grain_inds


# ---------------------------------------------------------------------------
# Periodic boundary conditions: prolongation matrix P
# Adapted from JAX-FEM applications/periodic_bc/example.py -- same P_mat
# mechanism, but node pairs come from Neper's $Periodicity section, and
# master chains (slave of a slave, occurring for edge/corner-type nodes
# paired in several directions) are resolved to root masters first.
# ---------------------------------------------------------------------------

def build_P_mat(num_nodes, vec, slaves, masters):
    # Direct master assignment; a node listed as slave several times keeps
    # the last entry -- roots agree because the lattice translations commute.
    master_of = onp.arange(num_nodes)
    master_of[slaves] = masters

    # Resolve chains to root masters.
    while True:
        new = master_of[master_of]
        if onp.array_equal(new, master_of):
            break
        master_of = new

    root_nodes = onp.where(master_of == onp.arange(num_nodes))[0]
    num_masters = len(root_nodes)

    col_of_node = -onp.ones(num_nodes, dtype=onp.int64)
    col_of_node[root_nodes] = onp.arange(num_masters)

    N = num_nodes * vec
    M = num_masters * vec
    I = onp.arange(N)
    # dof i = node*vec + comp  ->  column = col_of_node[root(node)]*vec + comp
    nodes = I // vec
    comps = I % vec
    J = col_of_node[master_of[nodes]] * vec + comps
    V = onp.ones(N)
    P_mat = scipy.sparse.csr_array((V, (I, J)), shape=(N, M))

    # Reduced-dof indices of the master (root) dofs, used for warm starts:
    # w_reduced = w_full[master_dofs].
    master_dofs = (root_nodes[:, None] * vec + onp.arange(vec)[None, :]).reshape(-1)
    return P_mat, master_of, master_dofs


# ---------------------------------------------------------------------------
# Fluctuation-formulation wrapper around the (unmodified) CP model.
# The macroscopic gradient H_bar is carried as a 5th internal variable so
# the jitted kernels receive it as a runtime argument (changing it per step
# or per macro iteration does not bake stale constants into the JIT cache).
# ---------------------------------------------------------------------------

class CrystalPlasticityPBC(CrystalPlasticity):
    def custom_init(self, quat, cell_ori_inds):
        super().custom_init(quat, cell_ori_inds)
        num_cells = len(self.fes[0].cells)
        H_bar_gp = onp.zeros((num_cells, self.fes[0].num_quads, self.dim, self.dim))
        self.internal_vars = self.internal_vars + [H_bar_gp]

    def get_tensor_map(self):
        base_tensor_map, _ = self.get_maps()

        def tensor_map(w_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, H_bar):
            # total displacement gradient = macro part + fluctuation part
            return base_tensor_map(w_grad + H_bar, Fp_inv_old, slip_resistance_old,
                                   slip_old, rot_mat)
        return tensor_map

    def _total_u_grads(self, sol, H_bar_gp):
        # identical gradient reconstruction to the base model, plus H_bar
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] \
            * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)
        return u_grads + H_bar_gp

    def update_int_vars_gp(self, sol, params):
        *cp_params, H_bar_gp = params
        _, update_int_vars_map = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        u_grads = self._total_u_grads(sol, H_bar_gp)
        Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp = \
            vmap_update_int_vars_map(u_grads, *cp_params)
        return [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp, H_bar_gp]

    def compute_avg_stress(self, sol, params):
        """Cell-averaged Cauchy stress, with the total deformation gradient."""
        *cp_params, H_bar_gp = params
        u_grads = self._total_u_grads(sol, H_bar_gp)

        partial_tensor_map, _ = self.get_maps()
        vmap_partial_tensor_map = jax.jit(jax.vmap(jax.vmap(partial_tensor_map)))
        P = vmap_partial_tensor_map(u_grads, *cp_params)

        def P_to_sigma(P, F):
            return 1. / np.linalg.det(F) * P @ F.T

        vvmap_P_to_sigma = jax.vmap(jax.vmap(P_to_sigma))
        F = u_grads + np.eye(self.dim)[None, None, :, :]
        sigma = vvmap_P_to_sigma(P, F)

        sigma_cell_data = np.sum(sigma * self.fes[0].JxW[:, :, None, None], 1) \
            / np.sum(self.fes[0].JxW, axis=1)[:, None, None]
        return sigma_cell_data


# ---------------------------------------------------------------------------
# Macro-control helpers
# ---------------------------------------------------------------------------

def elastic_macro_jacobian(C, slots):
    """Exact elastic dR/dtheta: R_a = <sigma>_(ia,ja), theta_b drives H_(kb,lb)
    (and H_(lb,kb) for shear slots). Used to seed the Broyden iteration."""
    n = len(slots)
    J = onp.zeros((n, n))
    for a, (i, j) in enumerate(slots):
        for b, (k, l) in enumerate(slots):
            J[a, b] = C[i, j, k, l] + (C[i, j, l, k] if k != l else 0.)
    return J


def von_mises(sig):
    return onp.sqrt(0.5 * ((sig[0, 0] - sig[1, 1])**2
                           + (sig[1, 1] - sig[2, 2])**2
                           + (sig[2, 2] - sig[0, 0])**2)
                    + 3.0 * (sig[0, 1]**2 + sig[1, 2]**2 + sig[0, 2]**2))


# ---------------------------------------------------------------------------
# The general driver: any MacroStep history, any prescribed/target mask.
# ---------------------------------------------------------------------------

def run_history(name, steps, num_steps=None, vtk_every_step=False):
    print("backend:", jax.default_backend())
    if num_steps is not None:
        steps = steps[:num_steps]
    print(f"history '{name}': {len(steps)} increments")

    tag = name.replace('-', '_')
    # Esi: per-step VTU series in the original example's folder layout
    # (data/vtk/<case>/..., one u_XXX.vtu per increment) for ParaView.
    vtk_series_dir = os.path.join(crt_dir, 'data', 'vtk',
                                  'polycrystal_304steel_pbc', tag)
    if vtk_every_step:
        os.makedirs(vtk_series_dir, exist_ok=True)
        for f in os.listdir(vtk_series_dir):
            os.remove(os.path.join(vtk_series_dir, f))

    # uniform dt required: jax-fem kernels bake problem.dt at first JIT trace
    ts = onp.array([0.] + [s.t for s in steps])
    dts = onp.diff(ts)
    assert onp.allclose(dts, dts[0]), "non-uniform dt is not supported " \
        "(problem.dt is baked into the jitted kernels at first trace)"

    msh_file = os.path.join(rve_dir, 'rve100.msh')
    tess_file = os.path.join(rve_dir, 'rve100.tess')

    points, cells, cell_grain_inds = read_msh_mesh(msh_file)
    quat = read_tess_quaternions(tess_file)
    slaves, masters, shifts = read_msh_periodicity(msh_file)
    print(f"mesh: {len(points)} nodes, {len(cells)} TET4 cells, "
          f"{quat.shape[0]} grains, {len(slaves)} periodic node pairs")

    ele_type = 'TET4'
    assert get_meshio_cell_type(ele_type) == 'tetra'
    mesh = Mesh(points, cells)

    vec, dim = 3, 3
    P_mat, master_of_node, master_dofs = build_P_mat(len(points), vec, slaves, masters)
    print(f"P_mat: {P_mat.shape[0]} full dofs -> {P_mat.shape[1]} reduced dofs")

    # Pin the fluctuation at one interior node (not in any periodic pair).
    paired = set(slaves.tolist()) | set(masters.tolist())
    unpaired = [i for i in range(len(points)) if i not in paired]
    center = onp.array([0.5, 0.5, 0.5])
    pin_node = unpaired[int(onp.argmin(onp.linalg.norm(points[unpaired] - center, axis=1)))]
    pin_point = points[pin_node]
    print(f"pinning fluctuation at node {pin_node}, X = {pin_point}")

    def pin(point):
        return np.all(np.isclose(point, pin_point, atol=1e-9))

    def zero_dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[pin] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]

    problem = CrystalPlasticityPBC(mesh, vec=vec, dim=dim, ele_type=ele_type,
                                   dirichlet_bc_info=dirichlet_bc_info,
                                   additional_info=(quat, cell_grain_inds))
    problem.P_mat = P_mat

    num_cells = len(problem.fes[0].cells)
    num_quads = problem.fes[0].num_quads
    cell_vols = onp.array(np.sum(problem.fes[0].JxW, axis=1))
    total_vol = cell_vols.sum()
    w_cell = np.array(cell_vols / total_vol)
    print(f"num_quads per cell: {num_quads}, total volume = {total_vol:.6f}")

    # committed internal state (Fp_inv, slip resistance, slip, rotations) --
    # mutated ONLY at the commit point at the end of each increment
    cp_params = problem.internal_vars[:4]
    w_red = None
    n_solves = 0

    def solve_with_H(H):
        """One FE solve at fixed macro gradient H, starting from the COMMITTED
        internal state cp_params. Returns fluctuation sol, cell stresses, and
        the volume-averaged stress tensor."""
        nonlocal w_red, n_solves
        H_bar_gp = onp.broadcast_to(H, (num_cells, num_quads, dim, dim)).copy()
       # print(H_bar_gp)
        problem.set_params(cp_params + [H_bar_gp])
        options = {'jax_solver': {}}
        if w_red is not None:
            options['initial_guess'] = w_red
        sol = solver(problem, solver_options=options)[0]
        n_solves += 1
        w_full = onp.array(sol).reshape(-1)
        w_red = np.array(w_full[master_dofs])
        sigma_cell = problem.compute_avg_stress(sol, problem.internal_vars)
        sig_avg = onp.array(np.sum(sigma_cell * w_cell[:, None, None], axis=0))
        return sol, sigma_cell, sig_avg

    rows, iter_rows = [], []
    theta_hist = []            # converged unknown vectors, for extrapolation
    slots_prev = None
    J = None
    sol_fluct = sigma_cell = None
    H = onp.zeros((dim, dim))

    for k, step in enumerate(steps):
        problem.dt = float(dts[0])
        slots = step.unknown_slots()
        slot_names = [f"H_{'xyz'[i]}{'xyz'[j]}" for (i, j) in slots]
        print(f"\nStep {k + 1} in {len(steps)}, t = {step.t:.4f}, "
              f"prescribed H_zz = {step.H_presc[2, 2]:.6f}, "
              f"unknowns: {slot_names if slots else 'none (fully prescribed)'}")

        if not slots:
            # fully prescribed F_bar: single solve
            H = step.assemble_H(onp.array([]))
            sol_fluct, sigma_cell, sig_avg = solve_with_H(H)
            n_iters, res_inf = 1, 0.
            theta_hist.append(onp.array([]))
        else:
            # ---- mixed control: Broyden iteration on theta -> stress targets
            targets = onp.array([step.sigma_targets[s] for s in slots])
            if slots_prev != slots or J is None:
                J = elastic_macro_jacobian(problem.C, slots)
                theta_hist = []          # slot set changed: restart history

            def presc_stress_estimate(H_presc, mask):
                """Elastic estimate of the stress the PRESCRIBED components
                induce on each target slot (predictor only)."""
                return onp.array([sum(problem.C[i, j, m, n] * H_presc[m, n]
                                      for m in range(3) for n in range(3)
                                      if mask[m, n])
                                  for (i, j) in slots])

            # initial guess for theta
            if len(theta_hist) == 0:
                # elastic predictor from scratch
                theta = onp.linalg.solve(J, targets - presc_stress_estimate(
                    step.H_presc, step.mask))
            elif len(theta_hist) == 1:
                # previous theta + tangent response to the prescribed increment
                dP = presc_stress_estimate(step.H_presc - step_prev.H_presc,
                                           step.mask)
                theta = theta_hist[-1] + onp.linalg.solve(J, -dP)
            else:
                # linear extrapolation along the loading path
                theta = 2. * theta_hist[-1] - theta_hist[-2]

            thetas, residuals = [], []   # ACCEPTED evaluations only
            converged = False
            d = None
            halvings_left = MACRO_MAX_HALVINGS
            n_iters = 0
            for it in range(MACRO_MAX_ITERS):
                H = step.assemble_H(theta)
                sol_fluct, sigma_cell, sig_avg = solve_with_H(H)
                R = onp.array([sig_avg[i, j] for (i, j) in slots]) - targets
                res_inf = float(onp.abs(R).max())
                n_iters = it + 1
                iter_rows.append([k + 1, it, res_inf] + theta.tolist() + R.tolist())
                print(f"  macro-iter {it}: |R|_inf = {res_inf:.4f} MPa, "
                      f"theta = {onp.array2string(theta, precision=7)}")

                if res_inf < MACRO_TOL:
                    converged = True
                    break

                # step-halving guard: if the residual grew markedly, retreat
                # half of the last Newton step (rejected points do not feed
                # the Broyden update)
                if (residuals and d is not None and halvings_left > 0
                        and res_inf > 1.5 * float(onp.abs(residuals[-1]).max())):
                    d = 0.5 * d
                    theta = thetas[-1] + d
                    halvings_left -= 1
                    continue

                # accept the point
                thetas.append(theta.copy())
                residuals.append(R.copy())
                if len(thetas) >= 2:
                    d_theta = thetas[-1] - thetas[-2]
                    d_R = residuals[-1] - residuals[-2]
                    denom = float(d_theta @ d_theta)
                    if denom > 0.:
                        J = J + onp.outer(d_R - J @ d_theta, d_theta) / denom

                d = onp.linalg.solve(J, -R)
                theta = theta + d
                halvings_left = MACRO_MAX_HALVINGS

            if not converged:
                print(f"  WARNING: macro iteration stopped at |R|_inf = "
                      f"{res_inf:.4f} MPa (> {MACRO_TOL} MPa) after {n_iters} solves")
            theta_hist.append(theta.copy())   # last EVALUATED point = committed state
        slots_prev = slots
        step_prev = step

        # ---- COMMIT: only now, after FE + macro convergence (path dependence)
        new_params = problem.update_int_vars_gp(sol_fluct, problem.internal_vars)
        cp_params = new_params[:4]

        # ---- record increment outputs
        F = H + onp.eye(dim)
        mean_stress = onp.trace(sig_avg) / 3.
        vm_avg_tensor = von_mises(sig_avg)
        s_vm_cell = np.sqrt(0.5 * ((sigma_cell[:, 0, 0] - sigma_cell[:, 1, 1])**2
                                   + (sigma_cell[:, 1, 1] - sigma_cell[:, 2, 2])**2
                                   + (sigma_cell[:, 2, 2] - sigma_cell[:, 0, 0])**2)
                            + 3.0 * (sigma_cell[:, 0, 1]**2 + sigma_cell[:, 1, 2]**2
                                     + sigma_cell[:, 0, 2]**2))
        avg_local_vm = float(np.sum(s_vm_cell * w_cell))
        rows.append([k + 1, step.t]
                    + F.reshape(-1).tolist()
                    + [sig_avg[0, 0], sig_avg[1, 1], sig_avg[2, 2],
                       sig_avg[0, 1], sig_avg[0, 2], sig_avg[1, 2],
                       mean_stress, vm_avg_tensor, avg_local_vm,
                       n_iters, res_inf])
        print(f"<sigma> [MPa]: xx = {sig_avg[0, 0]:.3f}, yy = {sig_avg[1, 1]:.3f}, "
              f"zz = {sig_avg[2, 2]:.3f}, xy = {sig_avg[0, 1]:.3f}, "
              f"xz = {sig_avg[0, 2]:.3f}, yz = {sig_avg[1, 2]:.3f} | "
              f"mean = {mean_stress:.3f}, vM(avg) = {vm_avg_tensor:.3f}, "
              f"avg(local vM) = {avg_local_vm:.3f}")

        if vtk_every_step:
            u_total_k = points @ H.T + onp.array(sol_fluct)
            save_sol(problem.fes[0], sol_fluct,
                     os.path.join(vtk_series_dir, f'u_{k:03d}.vtu'),
                     cell_infos=[('grain_id', onp.array(cell_grain_inds, dtype=onp.float64)),
                                 ('sigma_zz', onp.array(sigma_cell[:, 2, 2])),
                                 ('von_Mises_stress', onp.array(s_vm_cell))],
                     point_infos=[('u_total', u_total_k)])

    # ---- outputs ----
    header = ("step,t,F_xx,F_xy,F_xz,F_yx,F_yy,F_yz,F_zx,F_zy,F_zz,"
              "sig_xx,sig_yy,sig_zz,sig_xy,sig_xz,sig_yz,"
              "mean_stress,vm_of_avg_tensor,avg_local_vm,"
              "macro_iters,macro_res_inf_MPa")
    csv_path = os.path.join(results_dir, f'macro_{tag}.csv')
    onp.savetxt(csv_path, onp.array(rows), delimiter=",", header=header, comments="")
    print(f"\nSaved {csv_path}")

    if iter_rows:
        n_slots = (len(iter_rows[0]) - 3) // 2
        it_header = ("step,iter,res_inf_MPa,"
                     + ",".join(f"theta_{a}" for a in range(n_slots)) + ","
                     + ",".join(f"R_{a}_MPa" for a in range(n_slots)))
        iters_path = os.path.join(results_dir, f'macro_{tag}_iters.csv')
        onp.savetxt(iters_path, onp.array(iter_rows), delimiter=",",
                    header=it_header, comments="")
        print(f"Saved {iters_path}")
    print(f"total FE solves: {n_solves} for {len(steps)} increments")

    # final-step VTU (fields from the last converged solve)
    s_vm_cell = onp.array(s_vm_cell)
    u_total = points @ H.T + onp.array(sol_fluct)
    vtu_path = os.path.join(results_dir, f'u_final_{tag}.vtu')
    save_sol(problem.fes[0], sol_fluct, vtu_path,
             cell_infos=[('grain_id', onp.array(cell_grain_inds, dtype=onp.float64)),
                         ('sigma_zz', onp.array(sigma_cell[:, 2, 2])),
                         ('von_Mises_stress', s_vm_cell)],
             point_infos=[('u_total', u_total)])
    print(f"Saved {vtu_path}")

    # PBC sanity check: fluctuation must match on periodic node pairs
    w = onp.array(sol_fluct)
    err = onp.abs(w[slaves] - w[masters]).max()
    print(f"PBC check: max |w_slave - w_master| = {err:.3e}")

    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='uniaxial-stress',
                        choices=sorted(PRESETS.keys()),
                        help="loading history preset (see macro_loading.py)")
    parser.add_argument('--steps', type=int, default=None,
                        help="truncate the history to its first N increments")
    parser.add_argument('--vtk-every-step', action='store_true',
                        help="write a u_XXX.vtu per increment into "
                             "data/vtk/polycrystal_304steel_pbc/<preset>/ "
                             "(ParaView time series, original example layout)")
    args = parser.parse_args()

    t0 = time.time()
    run_history(args.preset, PRESETS[args.preset](args.steps),
                num_steps=args.steps, vtk_every_step=args.vtk_every_step)
    print(f"\nSimulation time: {time.time() - t0:.1f} s")
