#!/usr/bin/env python3
"""
One macroscopic point: the RVE driven directly by a prescribed deformation gradient,
with corner-based periodic boundary conditions.

Corner 0 (0, 0, 0)  fixed.
Corner 1 (L, 0, 0)  u_x, u_y, u_z loaded.
Corner 2 (0, L, 0)  u_x = 0; u_y, u_z loaded.
Corner 3 (0, 0, L)  u_x = u_y = 0; u_z loaded.

Each slave surface follows its master surface plus the matching corner displacement.
With u = (F - I) X at the corners, the 3 constrained dofs give H_xy = H_xz = H_yz = 0,
so F is lower triangular. Each of the 6 loaded dofs follows its own GP curve from
_gp_loading. The history of the RVE is kept between steps.

The screen shows the homogenized state at each step. Three files are written, with one
row per time step and each tensor row by row, ready for gnuplot:

    <prefix>_F.txt   F (9 columns), Cauchy stress (9 columns), dt
    <prefix>_U.txt   stretch U (9 columns), Cauchy stress (9 columns), dt
    <prefix>_R.txt   rotation R (9 columns), Cauchy stress (9 columns), dt

U and R come from the polar decomposition F = R U. After the run, rve_step_XX.png is
written per step: the deformed RVE, with each micro tet coloured by its own local stress.

    python single_point_gp.py --steps 5 --variance 1e-4 --dt 0.01 --component xx
"""

import argparse

import numpy as np

from auxFunctions import _create_loading_function, _gp_loading, _read_deformation_gradient
from rve_crystal_plasticity import Micromodel

# Independent components of the symmetric H
COMPONENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

# Order of the loaded dofs: (corner, displacement component)
LOADED_DOFS = [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2)]


def file_header(symbol):
    """Column names of one output file (gnuplot ignores lines starting with #)."""
    tensor = " ".join(f"{symbol}_{i}{j}" for i in "xyz" for j in "xyz")
    cauchy = " ".join(f"sig_{i}{j}" for i in "xyz" for j in "xyz")
    return f"# {tensor} {cauchy} dt\n"


def H_from_corner_dofs(u1_x, u1_y, u1_z, u2_y, u2_z, u3_z, L=1.0):
    """Macro gradient from the 6 loaded corner dofs. The columns are u1/L, u2/L and u3/L."""
    return np.array([[u1_x, 0.0, 0.0],
                     [u1_y, u2_y, 0.0],
                     [u1_z, u2_z, u3_z]]) / L


def gp_corner_history(n_steps, variance, lengthscale=20, seed=1):
    """H per step from one GP curve per loaded corner dof, each from 0 with its own seed."""
    curves = [_gp_loading(n_steps + 1, lengthscale=lengthscale, variance=variance, seed=seed + a)[1:]
              for a in range(len(LOADED_DOFS))]
    return np.array([H_from_corner_dofs(*values) for values in zip(*curves)])


def monotonic_corner_history(n_steps, final_dofs):
    """H per step, each loaded dof ramping linearly from 0 to its value in final_dofs."""
    ramp = _create_loading_function("monotonic", n_steps=n_steps + 1, start=0.0, end=1.0)[1:]
    return np.array([H_from_corner_dofs(*(factor * np.asarray(final_dofs))) for factor in ramp])


def polar_decomposition(F):
    """Rotation R and symmetric stretch U of F = R U."""
    eigenvalues, eigenvectors = np.linalg.eigh(F.T @ F)
    U = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    return F @ np.linalg.inv(U), U


def gp_stretch_history(n_steps, variance, lengthscale=20, seed=1):
    """H per step, shape (n_steps, 3, 3). Each component has its own seed and starts from 0."""
    H = np.zeros((n_steps, 3, 3))
    for a, (i, j) in enumerate(COMPONENTS):
        curve = _gp_loading(n_steps + 1, lengthscale=lengthscale, variance=variance, seed=seed + a)[1:]
        H[:, i, j] = curve
        H[:, j, i] = curve
    return H


def report_step(step, t, dt, F, P, files):
    """Prints the homogenized state of one step on the screen, and adds a row to each file."""
    sigma = P @ F.T / np.linalg.det(F)
    R, U = polar_decomposition(F)
    print(f"Step {step}, t = {t:.6f}, dt = {dt:.6f}\n"
          f"Homogenized deformation gradient F (prescribed):\n{np.array2string(F, precision=8, suppress_small=True)}\n"
          f"Homogenized stretch U:\n{np.array2string(U, precision=8, suppress_small=True)}\n"
          f"Homogenized rotation R:\n{np.array2string(R, precision=8, suppress_small=True)}\n"
          f"Homogenized first Piola-Kirchhoff stress P [MPa]:\n{np.array2string(P, precision=4, suppress_small=True)}\n"
          f"Homogenized Cauchy stress [MPa]:\n{np.array2string(sigma, precision=4, suppress_small=True)}\n",
          flush=True)
    for symbol, tensor in (("F", F), ("U", U), ("R", R)):
        files[symbol].write(" ".join(f"{value:.10e}" for value in np.r_[tensor.ravel(), sigma.ravel(), dt]) + "\n")
        files[symbol].flush()


def plot_rve_steps(rve, H_history, w_history, stress_history, component="xx", clim=None, warp=20.0):
    """
    Writes rve_step_XX.png per step: the deformed RVE, with each micro tet coloured by
    its own local P, not by the homogenized stress. The box marks the undeformed RVE.

    All steps share one colour range, so colours can be compared between them. By
    default it is the minimum and maximum of the whole run. `warp` magnifies the
    displacement.
    """
    import pyvista as pv

    n_tets = rve.cells.shape[0]
    cells = np.column_stack([np.full(n_tets, 4), rve.cells]).ravel()
    undeformed = pv.UnstructuredGrid(cells, np.full(n_tets, pv.CellType.TETRA), rve.points)
    limits = list(clim) if clim is not None else [stress_history.min(), stress_history.max()]
#    limits = list(clim) if clim is not None else list(np.percentile(stress_history, [1, 99]))
    name = f"Local P_{component} [MPa]"

    for k, (H, w, stress) in enumerate(zip(H_history, w_history, stress_history)):
        deformed = undeformed.copy()
        deformed.points = rve.points + warp * rve.total_displacement(H, w)
        deformed.cell_data[name] = stress

        plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
        plotter.add_mesh(deformed, scalars=name, cmap="coolwarm", clim=limits, show_edges=True,
                         edge_color="black", line_width=0.3, lighting=False)
        plotter.add_mesh(undeformed.outline(), color="black", line_width=1)
        plotter.add_text(f"Step {k + 1}, displacement x {warp:g}\n"
                         f"min {stress.min():.0f}, max {stress.max():.0f} MPa", font_size=11)
        plotter.view_isometric()
        plotter.screenshot(f"rve_step_{k + 1:02d}.png")
        plotter.close()


def run(H_history, dt, rve=None, prefix="single_point_gp", component="xx", plot=None):
    """
    Runs the RVE through H_history and writes the <prefix>_F, _U and _R files. Returns
    (F, P) per step. `component` is the local stress to plot, such as "xx" or "yz", and
    `plot` holds keyword arguments of plot_rve_steps ({} uses the defaults, None skips
    the pictures).
    """
    rve = rve if rve is not None else Micromodel()
    i, j = ("xyz".index(component[0]), "xyz".index(component[1]))
    w_history, stress_history = [], []
    history, w = rve.initial_history(), None
    H_last, dw_dH, results = None, None, []

    files = {symbol: open(f"{prefix}_{symbol}.txt", "w") for symbol in "FUR"}
    for symbol, file in files.items():
        file.write(file_header(symbol))

    for k, H in enumerate(H_history):
        # First-order prediction of the fluctuation from the last solution
        w_init = None if w is None else w + dw_dH @ (H - H_last).ravel()
        ok, P, _, w, history = rve.evaluate_homogenized_properties(H, history, dt, w_init)
        if not ok:
            raise RuntimeError(f"The RVE did not converge at step {k + 1}. Try a smaller amplitude or dt.")
        H_last, dw_dH = H, rve.dw_dH.copy()

        F = np.eye(3) + H
        report_step(k + 1, (k + 1) * dt, dt, F, P, files)
        results.append((F, P.copy()))
        w_history.append(w.copy())
        stress_history.append(rve.P[:, i, j].copy())

    for file in files.values():
        file.close()

    if plot is not None:
        plot_rve_steps(rve, H_history, w_history, np.array(stress_history), component, **plot)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading", default="gp", choices=["gp", "monotonic", "file"], help="type of loading")
    parser.add_argument("--file", default="/results/macro_uniaxial_stress.csv",
                        help="csv file with the full F, for --loading file")
    parser.add_argument("--final-dofs", type=float, nargs=6, default=[0.002, 0.0, 0.0, 0.0, 0.0, 0.0],
                        metavar=("U1X", "U1Y", "U1Z", "U2Y", "U2Z", "U3Z"),
                        help="final values of the 6 loaded corner dofs, for monotonic loading")
    parser.add_argument("--steps", type=int, default=40, help="number of time steps")
    parser.add_argument("--variance", type=float, default=1e-6, help="GP variance, sets the amplitude")
    parser.add_argument("--lengthscale", type=float, default=20, help="GP lengthscale, in steps")
    parser.add_argument("--dt", type=float, default=0.01, help="time step (the model is rate-dependent)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the first component")
    parser.add_argument("--prefix", default="single_point_gp", help="start of the output file names")
    parser.add_argument("--component", default="xx", help="local stress to plot: xx, xy, ...")
    parser.add_argument("--clim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                        help="colour limits [MPa], shared by all steps")
    parser.add_argument("--warp", type=float, default=20.0, help="displacement magnification")
    parser.add_argument("--no-plots", action="store_true", help="skip rve_step_XX.png")
    args = parser.parse_args()

    dt = args.dt
    if args.loading == "file":
        H_history, file_dt = _read_deformation_gradient(args.file)
        dt = file_dt if file_dt is not None else dt
    elif args.loading == "monotonic":
        H_history = monotonic_corner_history(args.steps, args.final_dofs)
    else:
        H_history = gp_corner_history(args.steps, args.variance, args.lengthscale, args.seed)

    run(H_history, dt, prefix=args.prefix, component=args.component,
        plot=None if args.no_plots else {"clim": args.clim, "warp": args.warp})