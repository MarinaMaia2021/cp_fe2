"""3D macro cube at finite strain, with displacements prescribed as u = H X."""

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh


class Macromodel3D:
    def __init__(self, divisions=(1, 1, 1), size=(1.0, 1.0, 1.0), load_case="uniaxial"):
        self.divisions = divisions
        self.L, self.Ly, self.Lz = size   # L: length along x, the loading direction
        self.load_case = load_case

        self.domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], list(size)], list(divisions),
                                      mesh.CellType.tetrahedron, ghost_mode=mesh.GhostMode.shared_facet)
        self.domain.topology.create_connectivity(2, 3)

        self.V = fem.functionspace(self.domain, ("P", 1, (3,)))
        self.u = fem.Function(self.V, name="Macro_Displacement")
        self.u_test = ufl.TestFunction(self.V)
        self.u_trial = ufl.TrialFunction(self.V)

        # Macro fields, one value per cell (one RVE per tet)
        self.W_grad = fem.functionspace(self.domain, ("DG", 0, (3, 3)))
        self.stress_field = fem.Function(self.W_grad, name="Macro_P")
        self.W_tangent = fem.functionspace(self.domain, ("DG", 0, (9, 9)))
        self.tangent_field = fem.Function(self.W_tangent, name="Macro_Tangent")

        self._setup_bc()

    def _constrained_nodes(self):
        """For each displacement component, a function selecting its constrained nodes."""
        L, Ly, Lz = self.L, self.Ly, self.Lz
        left = lambda x: np.isclose(x[0], 0.0)
        right = lambda x: np.isclose(x[0], L)
        ends = lambda x: left(x) | right(x)
        origin = lambda x: left(x) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0)
        point_y = lambda x: left(x) & np.isclose(x[1], Ly) & np.isclose(x[2], 0.0)
        boundary = lambda x: ends(x) | np.isclose(x[1], 0.0) | np.isclose(x[1], Ly) | np.isclose(x[2], 0.0) | np.isclose(x[2], Lz)

        if self.load_case == "uniaxial":
            # Left face in x, plus the three dofs that stop rigid motion in the y-z plane
            return [ends, origin, lambda x: origin(x) | point_y(x)]
        if self.load_case == "uniaxial_shear":
            # As uniaxial, and the right face also follows H_zx in z
            return [ends, origin, lambda x: origin(x) | point_y(x) | right(x)]
        if self.load_case == "clamped":
            # Left face fully fixed, right face pulled along x and free in y and z.
            # The macro deformation is not uniform.
            return [ends, left, left]
        if self.load_case == "prescribed":
            # Fully prescribed H_bar: u = H X on the whole boundary
            return [boundary, boundary, boundary]
        raise ValueError(f"Unknown load case '{self.load_case}'")

    def _setup_bc(self):
        self.bcs = []
        self.bc_values = []
        for i, selector in enumerate(self._constrained_nodes()):
            V_i, _ = self.V.sub(i).collapse()
            g = fem.Function(V_i)
            dofs = fem.locate_dofs_geometrical((self.V.sub(i), V_i), selector)
            self.bcs.append(fem.dirichletbc(g, dofs, self.V.sub(i)))
            self.bc_values.append(g)

    def apply_load(self, H):
        """Boundary values u_i = H_ij X_j on the constrained nodes."""
        for i, g in enumerate(self.bc_values):
            g.interpolate(lambda x: H[i, 0] * x[0] + H[i, 1] * x[1] + H[i, 2] * x[2])

    def weak_forms(self):
        dx = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": 1})
        as_vec = lambda t: ufl.as_vector([t[i, j] for i in range(3) for j in range(3)])

        F_form = ufl.inner(self.stress_field, ufl.grad(self.u_test)) * dx
        J_form = ufl.inner(ufl.dot(self.tangent_field, as_vec(ufl.grad(self.u_trial))),
                           as_vec(ufl.grad(self.u_test))) * dx

        # Reaction at x = L as virtual work of a unit x displacement of that face
        w = fem.Function(self.V)
        V_x, _ = self.V.sub(0).collapse()
        right_dofs = fem.locate_dofs_geometrical((self.V.sub(0), V_x), lambda x: np.isclose(x[0], self.L))[0]
        w.x.array[right_dofs] = 1.0
        w.x.scatter_forward()
        reaction_form = fem.form(ufl.inner(self.stress_field, ufl.grad(w)) * dx)
        return F_form, J_form, reaction_form

    def reaction_force(self, reaction_form):
        return self.domain.comm.allreduce(fem.assemble_scalar(reaction_form), op=MPI.SUM)
