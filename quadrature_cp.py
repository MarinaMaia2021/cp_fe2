"""Links macro cells to the RVE: strain, stress, tangent and per-cell micro history."""

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem


class MacroQuadratureMap:
    def __init__(self, macromodel, master_rve, verbose=False):
        self.macro_domain = macromodel.domain
        self.comm = self.macro_domain.comm
        self.master_rve = master_rve
        self.u_macro = macromodel.u
        self.stress_field = macromodel.stress_field
        self.tangent_field = macromodel.tangent_field
        self.verbose = verbose
        self.dt = 0.0

        tdim = self.macro_domain.topology.dim
        self.n_cells = self.macro_domain.topology.index_map(tdim).size_local

        self.grad_expr = fem.Expression(ufl.grad(self.u_macro), macromodel.W_grad.element.interpolation_points)
        self.grad_field = fem.Function(macromodel.W_grad)

        # One committed history per macro cell. Arrays are replaced, never modified,
        # so that keeping references in the checkpoint is enough.
        self.history = [master_rve.initial_history() for _ in range(self.n_cells)]
        self.history_trial = list(self.history)
        self.w_predictor = [None] * self.n_cells   # (H, w, dw/dH) of the latest solution
        self.store_micro_stress = False            # True keeps P per RVE, for plots
        self.P_micro = [None] * self.n_cells
        self.checkpoint()

    def update(self, u_macro_current=None):
        self.grad_field.interpolate(self.grad_expr)
        H_cells = self.grad_field.x.array[:9 * self.n_cells].reshape(self.n_cells, 3, 3)

        all_converged = True
        for c in range(self.n_cells):
            # First-order prediction of w from the sensitivities of the last solution
            w_init = None
            if self.w_predictor[c] is not None:
                H_last, w_last, dw_dH = self.w_predictor[c]
                w_init = w_last + dw_dH @ (H_cells[c] - H_last).ravel()
            ok, P, A, w, new_history = self.master_rve.evaluate_homogenized_properties(
                H_cells[c], self.history[c], self.dt, w_init)
            if not ok:
                all_converged = False
                break
            self.stress_field.x.array[9 * c : 9 * (c + 1)] = P.ravel()
            self.tangent_field.x.array[81 * c : 81 * (c + 1)] = A.ravel()
            self.history_trial[c] = new_history
            self.w_predictor[c] = (H_cells[c].copy(), w, self.master_rve.dw_dH.copy())
            if self.store_micro_stress:
                self.P_micro[c] = self.master_rve.P.copy()

        self.stress_field.x.scatter_forward()
        self.tangent_field.x.scatter_forward()
        return self.comm.allreduce(all_converged, op=MPI.LAND)

    def advance(self):
        """Commits the converged state of the current (sub-)step."""
        self.history = list(self.history_trial)

    def checkpoint(self):
        """Stores the state at the start of a macro step, before any sub-step."""
        self._checkpoint = (self.u_macro.x.array.copy(), list(self.history), list(self.w_predictor),
                            self.stress_field.x.array.copy(), self.tangent_field.x.array.copy())

    def rollback(self):
        """Returns to the last checkpoint, which also undoes committed sub-steps."""
        u, history, w_predictor, stress, tangent = self._checkpoint
        self.u_macro.x.array[:] = u
        self.stress_field.x.array[:] = stress
        self.tangent_field.x.array[:] = tangent
        self.u_macro.x.scatter_forward()
        self.history = list(history)
        self.history_trial = list(history)
        self.w_predictor = list(w_predictor)
