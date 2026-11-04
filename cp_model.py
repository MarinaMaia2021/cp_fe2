"""
Crystal plasticity at one integration point, in pure JAX.

The equations and parameters are those of models_304steel.py (Kalidindi-type
hardening, implicit S-based update). Differences, none of which change the physics:

  * dt is a runtime argument, so the step size can vary (needed for sub-stepping).
  * The inner Newton loop has an iteration limit and reports convergence.
  * NaN counts as "worse" in the line search, so it can no longer be accepted.
  * Rotated elastic and Schmid tensors are precomputed per grain.
  * The inner solve starts from a given S instead of always from zero (faster, and it
    converges to the same tolerance).
  * The tangent comes from one differentiable Newton step at the converged S, which
    is the implicit function theorem written without a custom_jvp.

Displacement gradient H = F - I. Stress returned: first Piola-Kirchhoff P.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

MAX_NEWTON = 60        # inner Newton iterations
MAX_LINE_SEARCH = 8    # step halvings per Newton iteration
INNER_TOL = 1e-8       # |S - C:E| [MPa]


class CPParams(NamedTuple):
    q: jnp.ndarray     # latent hardening matrix
    h: float = 392.9772       # initial hardening [MPa]
    t_sat: float = 7295.1754  # saturation strength [MPa]
    gss_a: float = 8.0        # hardening exponent
    ao: float = 0.001         # reference strain rate
    xm: float = 1.0 / 120.0   # rate sensitivity exponent


GSS_INITIAL = 90.0     # initial slip resistance [MPa]


def rotation_matrices(quats):
    """Rotation matrices from scalar-first quaternions (same formula as get_rot_mat)."""
    q0, q1, q2, q3 = np.asarray(quats).T
    return np.stack([
        q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2,
        2*q1*q2 + 2*q0*q3, q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*q2*q3 - 2*q0*q1,
        2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0*q0 - q1*q1 - q2*q2 + q3*q3,
    ], axis=1).reshape(-1, 3, 3)


def cubic_stiffness(C11=2.622e5, C12=1.120e5, C44=0.746e5):
    """Cubic elastic tensor C_ijkl [MPa], crystal axes."""
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        C[i, i, i, i] = C11
        for j in range(3):
            if i != j:
                C[i, i, j, j] = C12
                C[i, j, i, j] = C[i, j, j, i] = C44
    return C


def schmid_tensors(slip_sys_file):
    """Schmid tensors from a file with one 'normal direction' row per slip system."""
    data = np.loadtxt(slip_sys_file)
    normals = data[:, :3] / np.linalg.norm(data[:, :3], axis=1, keepdims=True)
    directions = data[:, 3:] / np.linalg.norm(data[:, 3:], axis=1, keepdims=True)
    return np.einsum("si,sj->sij", directions, normals)


def rotated_tensors(rot_mats, C, schmid):
    """Elastic and Schmid tensors in the sample frame, one entry per grain."""
    C_rot = np.einsum("gia,gjb,gkc,gld,abcd->gijkl", rot_mats, rot_mats, rot_mats, rot_mats, C, optimize=True)
    schmid_rot = np.einsum("gia,gjb,sab->gsij", rot_mats, rot_mats, schmid, optimize=True)
    return C_rot, schmid_rot


def _plastic_update(F, Fp_inv_old, g_old, slip_old, schmid, S, dt, p):
    """Slip, hardening and Fe for a trial S (`helper` in models_304steel.py)."""
    tau = jnp.sum(S[None, :, :] * schmid, axis=(1, 2))
    gamma_inc = p.ao * dt * jnp.abs(tau / g_old) ** (1.0 / p.xm) * jnp.sign(tau)

    tmp = p.h * jnp.abs(gamma_inc) * jnp.abs(1 - g_old / p.t_sat) ** p.gss_a * jnp.sign(1 - g_old / p.t_sat)
    g_new = g_old + p.q @ tmp
    slip_new = slip_old + gamma_inc

    L_plastic_inc = jnp.sum(gamma_inc[:, None, None] * schmid, axis=0)
    Fp_inv_new = Fp_inv_old @ (jnp.eye(3) - L_plastic_inc)
    return Fp_inv_new, g_new, slip_new, F @ Fp_inv_new


def _residual(y, F, Fp_inv_old, g_old, slip_old, C, schmid, dt, p):
    S = y.reshape(3, 3)
    *_, Fe = _plastic_update(F, Fp_inv_old, g_old, slip_old, schmid, S, dt, p)
    E = 0.5 * (Fe.T @ Fe - jnp.eye(3))
    return (S - jnp.sum(C * E[None, None, :, :], axis=(2, 3))).reshape(-1)


def _solve_S(y0, F, Fp_inv_old, g_old, slip_old, C, schmid, dt, p):
    """Inner Newton with the halving line search of models_304steel.py, from the guess y0."""
    res_fn = lambda y: _residual(y, F, Fp_inv_old, g_old, slip_old, C, schmid, dt, p)
    jac_fn = jax.jacfwd(res_fn)

    def not_done(state):
        step, res, _ = state
        return (step < MAX_NEWTON) & ~(jnp.linalg.norm(res) <= INNER_TOL)

    def newton(state):
        step, res, y = state
        y_inc = jnp.linalg.solve(jac_fn(y), -res)

        def keep_halving(ls):
            _, crt_res, sub = ls
            return ~(jnp.linalg.norm(crt_res) < jnp.linalg.norm(res)) & (sub < MAX_LINE_SEARCH)

        def halve(ls):
            relax, _, sub = ls
            return 0.5 * relax, res_fn(y + relax * y_inc), sub + 1

        relax, new_res, _ = jax.lax.while_loop(keep_halving, halve, (1.0, res, 0))
        return step + 1, new_res, y + 2.0 * relax * y_inc

    _, res, y = jax.lax.while_loop(not_done, newton, (0, res_fn(y0), y0))
    return y, jnp.linalg.norm(res) <= INNER_TOL


def _point_stress(H, S_guess, Fp_inv_old, g_old, slip_old, C, schmid, dt, p):
    F = H + jnp.eye(3)
    y_conv, converged = _solve_S(S_guess.reshape(-1), jax.lax.stop_gradient(F), Fp_inv_old, g_old, slip_old, C, schmid, dt, p)

    # One Newton step from the converged point carries the exact derivative dS/dH
    res_fn = lambda y, F_: _residual(y, F_, Fp_inv_old, g_old, slip_old, C, schmid, dt, p)
    J = jax.lax.stop_gradient(jax.jacfwd(res_fn, argnums=0)(y_conv, F))
    y = y_conv - jnp.linalg.solve(J, res_fn(y_conv, F))

    S = y.reshape(3, 3)
    Fp_inv_new, g_new, slip_new, Fe = _plastic_update(F, Fp_inv_old, g_old, slip_old, schmid, S, dt, p)
    sigma = Fe @ S @ Fe.T / jnp.linalg.det(Fe)
    P = jnp.linalg.det(F) * sigma @ jnp.linalg.inv(F).T
    return P, (P, Fp_inv_new, g_new, slip_new, converged, S)


# P, dP/dH and the updated history for a batch of points.
# S_guess only speeds up the inner solve: zeros are always a valid guess.
# Returns (A, (P, Fp_inv, g, slip, converged, S)), with A of shape (n, 3, 3, 3, 3).
update_and_tangent_batched = jax.jit(jax.vmap(
    jax.jacfwd(_point_stress, argnums=0, has_aux=True),
    in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
))
