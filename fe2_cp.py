#!/usr/bin/env python3
"""
FE2 with a 3D crystal plasticity RVE: finite-strain cube, one RVE per macroscopic tet, 
and tangents from autodiff.

Loading comes from macro_loading.py. Its histories load along z at a single point;
here the load acts on the right face (x = L), so x and z are swapped.
"""

import numpy as np
from mpi4py import MPI

from customnewtonproblem import CustomNewtonProblem
from macromodel_3d import Macromodel3D
from quadrature_cp import MacroQuadratureMap
from rve_crystal_plasticity import Micromodel
import macro_loading

SWAP_X_Z = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def load_case_of(steps):
    """Macro boundary conditions matching a macro_loading history."""
    if all(step.mask.all() for step in steps):
        return "prescribed"
    has_shear = any(abs(step.H_presc[0, 2]) > 0.0 for step in steps)
    return "uniaxial_shear" if has_shear else "uniaxial"


def macro_gradients(steps):
    """Prescribed H and times, moved to the macro axes (load along x)."""
    return [SWAP_X_Z @ step.H_presc @ SWAP_X_Z for step in steps], [step.t for step in steps]


def try_macro_step(problem, qmap, macromodel, H_prev, H_target, dt, substeps, is_root):
    """Advances one macro step with `substeps` equal increments, from the checkpoint."""
    qmap.rollback()
    qmap.dt = dt / substeps
    total_its = 0
    for k in range(1, substeps + 1):
        macromodel.apply_load(H_prev + (H_target - H_prev) * k / substeps)
        converged, its = problem.solve()
        total_its += its
        if not converged:
            if is_root:
                print(f"    Sub-step {k}/{substeps} failed to converge.", flush=True)
            qmap.rollback()
            return False, total_its
        qmap.advance()
    return True, total_its


def plot_load_displacement(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.r_[0.0, result["u"]], np.r_[0.0, result["force"]], "-o", color="#2a6fbb", markersize=4, linewidth=1.8)
    ax.set_xlabel("Displacement of the right face [mm]")
    ax.set_ylabel("Reaction force at the right face [N]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_stress_field(macromodel, qmap, step_index, component=(0, 0), clim=(0.0, 400.0), warp=20.0):
    """
    Writes macro_stress_XX.png: one colour per tet for a component of P, on the deformed
    mesh, with the undeformed mesh as a wireframe. The colour limits are fixed so that
    colours can be compared between steps, and `warp` magnifies the displacement.
    Works in parallel: every rank sends its own cells to rank 0.
    """
    V, W = macromodel.V, macromodel.W_grad
    n_cells, comm = qmap.n_cells, macromodel.domain.comm
    dofs = V.dofmap.list[:n_cells]
    X = V.tabulate_dof_coordinates()[dofs]
    u = macromodel.u.x.array.reshape(-1, 3)[dofs]
    stress_dofs = W.dofmap.list[:n_cells, 0]
    stress = macromodel.stress_field.x.array[9 * stress_dofs + 3 * component[0] + component[1]]

    gathered = comm.gather((X, u, stress), root=0)
    if comm.Get_rank() != 0:
        return

    import pyvista as pv
    X = np.concatenate([g[0] for g in gathered]).reshape(-1, 3)
    u = np.concatenate([g[1] for g in gathered]).reshape(-1, 3)
    stress = np.concatenate([g[2] for g in gathered])

    # Each tet keeps its own 4 points, so a cell-wise colour needs no shared numbering
    cells = np.column_stack([np.full(stress.size, 4), np.arange(4 * stress.size).reshape(-1, 4)]).ravel()
    undeformed = pv.UnstructuredGrid(cells, np.full(stress.size, pv.CellType.TETRA), X)
    deformed = undeformed.copy()
    deformed.points = X + warp * u
    name = f"P_{'xyz'[component[0]]}{'xyz'[component[1]]} [MPa]"
    deformed.cell_data[name] = stress

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
    # No lighting, so that the colours on screen match the colour bar
    plotter.add_mesh(deformed, scalars=name, cmap="coolwarm", clim=list(clim), show_edges=True,
                     edge_color="darkred", line_width=1, opacity=0.85, lighting=False)
    plotter.add_mesh(undeformed, style="wireframe", color="black", line_width=1)
    plotter.add_text(f"Step {step_index + 1}, displacement x {warp:g}", font_size=11)
    plotter.view_isometric()
    plotter.reset_camera(bounds=undeformed.bounds)
    plotter.screenshot(f"macro_stress_{step_index:02d}.png")
    plotter.close()


def plot_rve_stress(macromodel, qmap, step_index, rve_ids="all", component=(0, 0), clim=(0.0, 500.0), warp=20.0):
    """
    Writes rve_III_step_XX.png, one per time step and per RVE: the deformed RVE mesh,
    with each micro tet coloured by a component of its own local P. These are the
    microscopic stresses, not the homogenized ones. The box marks the undeformed RVE.

    rve_ids is "all" or a list of macro cell numbers (0-based, one RVE per macro tet).
    By default each image uses its own colour range (1st to 99th percentile), so that
    the local field is visible. Pass clim=(low, high) to compare colours between steps
    or RVEs. `warp` magnifies the displacement. In parallel, each rank plots its own RVEs.
    """
    import pyvista as pv

    rve = qmap.master_rve
    first_id = macromodel.domain.topology.index_map(3).local_range[0]
    n_tets = rve.cells.shape[0]
    cells = np.column_stack([np.full(n_tets, 4), rve.cells]).ravel()
    undeformed = pv.UnstructuredGrid(cells, np.full(n_tets, pv.CellType.TETRA), rve.points)
    name = f"Local P_{'xyz'[component[0]]}{'xyz'[component[1]]} [MPa]"

    for c in range(qmap.n_cells):
        rve_id = first_id + c
        if rve_ids != "all" and rve_id not in rve_ids:
            continue
        H, w, _ = qmap.w_predictor[c]
        stress = qmap.P_micro[c][:, component[0], component[1]]
        deformed = undeformed.copy()
        deformed.points = rve.points + warp * rve.total_displacement(H, w)
        deformed.cell_data[name] = stress
        limits = list(clim) if clim is not None else list(np.percentile(stress, [1, 99]))

        plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
        plotter.add_mesh(deformed, scalars=name, cmap="coolwarm", clim=limits, show_edges=True,
                         edge_color="black", line_width=0.3, lighting=False)
        plotter.add_mesh(undeformed.outline(), color="black", line_width=1)
        plotter.add_text(f"RVE {rve_id}, step {step_index + 1}, displacement x {warp:g}\n"
                         f"min {stress.min():.0f}, max {stress.max():.0f} MPa", font_size=11)
        plotter.view_isometric()
        plotter.screenshot(f"rve_{rve_id:03d}_step_{step_index:02d}.png")
        plotter.close()


def main(steps, divisions=(1, 1, 1), size=(1.0, 1.0, 1.0), load_case=None, rve=None,
         max_substeps=3, output=None, plot=None, stress_plot=None, rve_plot=None):
    """
    Runs a macro_loading history on a box of the given size, split into 6 tets per
    division. Returns a dict with time, displacement at x = L, reaction force at x = L,
    and Newton iterations per step. `output` and `plot` are rewritten after every step,
    so an interrupted run keeps its results. `stress_plot` holds keyword arguments of
    plot_stress_field, and turns those per-step pictures on ({} uses the defaults).
    `rve_plot` does the same for plot_rve_stress ({} plots all RVEs).
    """
    load_case = load_case or load_case_of(steps)
    macromodel = Macromodel3D(divisions=divisions, size=size, load_case=load_case)
    is_root = macromodel.domain.comm.Get_rank() == 0
    rve = rve if rve is not None else Micromodel()

    qmap = MacroQuadratureMap(macromodel, rve)
    qmap.store_micro_stress = rve_plot is not None
    F_form, J_form, reaction_form = macromodel.weak_forms()
    problem = CustomNewtonProblem(quadrature_map=qmap, directSolver=True, F=F_form, J=J_form,
                                  u=macromodel.u, bcs=macromodel.bcs)

    # Initial tangent for the first predictor
    gradients, times = macro_gradients(steps)
    qmap.dt = times[0]
    qmap.update()
    qmap.checkpoint()

    result = {"t": [], "u": [], "force": [], "its": []}
    H_prev, t_prev = np.zeros((3, 3)), 0.0
    for k, (H, t) in enumerate(zip(gradients, times)):
        if is_root:
            print(f"\n--- STEP {k} | t = {t:.4f} | H_xx = {H[0, 0]:.6f} ---", flush=True)

        qmap.checkpoint()
        for substeps in range(1, max_substeps + 1):
            converged, its = try_macro_step(problem, qmap, macromodel, H_prev, H, t - t_prev, substeps, is_root)
            if converged:
                break
        if not converged:
            if is_root:
                print(f"[FATAL] Macro step {k} failed with {max_substeps} sub-steps.", flush=True)
            break

        H_prev, t_prev = H, t
        force = macromodel.reaction_force(reaction_form)
        if stress_plot is not None:
            plot_stress_field(macromodel, qmap, k, **stress_plot)
        if rve_plot is not None:
            plot_rve_stress(macromodel, qmap, k, **rve_plot)
        if is_root:
            print(f"  Converged in {its} iterations. Reaction force: {force:.4f} N", flush=True)
            result["t"].append(t)
            result["u"].append(H[0, 0] * macromodel.L)
            result["force"].append(force)
            result["its"].append(its)
            if output:
                np.savetxt(output, np.column_stack([result[key] for key in ("t", "u", "force", "its")]),
                           header="t u_right force newton_iterations")
            if plot:
                plot_load_displacement(result, plot)

    return {key: np.array(val) for key, val in result.items()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20, help="steps of the monotonic ramp")
    parser.add_argument("--strain", type=float, default=0.002, help="final mean strain along x")
    parser.add_argument("--time", type=float, default=0.1, help="total time (the model is rate-dependent)")
    parser.add_argument("--size", type=float, nargs=3, default=[4.0, 1.0, 1.0], help="domain size")
    parser.add_argument("--divisions", type=int, nargs=3, default=[2, 1, 1], help="6 tets per division")
    parser.add_argument("--no-stress-plots", action="store_true", help="skip macro_stress_XX.png")
    parser.add_argument("--rves", nargs="+", default=["all"], help="RVE ids to plot, or all")
    parser.add_argument("--no-rve-plots", action="store_true", help="skip rve_III_step_XX.png")
    parser.add_argument("--rve-clim", type=float, nargs=2, default=[0.0, 100.0], metavar=("LOW", "HIGH"),
                    help="colour limits shared by all RVE plots [MPa]")
    args = parser.parse_args()

    # Left face clamped, right face pulled with displacement control
    ramp = macro_loading.uniaxial_stress_path(eps_max=args.strain, n_steps=args.steps, t_total=args.time)
    main(ramp, divisions=tuple(args.divisions), size=tuple(args.size), load_case="clamped",
         output="load_displacement.txt", plot="load_displacement.png",
         stress_plot=None if args.no_stress_plots else {},
         rve_plot=None if args.no_rve_plots else {"rve_ids": "all" if args.rves == ["all"] else [int(i) for i in args.rves]})