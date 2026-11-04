# Esi: load-path & macroscopic-control interface for the PBC CPFEM driver.
#
# A loading history is a list of MacroStep objects. Each step carries, for the
# macroscopic displacement-gradient tensor H_bar = F_bar - I:
#
#   * H_presc : (3,3) float -- component values used where `mask` is True
#   * mask    : (3,3) bool  -- True  = H_bar_ij is PRESCRIBED at the value above
#                              False = H_bar_ij is an UNKNOWN of the macro
#                                      stress-control iteration
#   * sigma_targets : {(i,j): float} -- target volume-averaged Cauchy stress
#                     for every unknown slot
#
# Unknown slots live on the symmetric set
#     [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)] :
# an off-diagonal unknown (i,j) drives H_ij AND H_ji with the same scalar
# theta (a symmetric stretch). The skew (rotation) part of H_bar is therefore
# always prescribed, which keeps rigid rotations out of the unknown space and
# makes the macro Newton system well posed.
#
# The driver in problem.py consumes any such history without knowing how it
# was built; the builders below are convenience constructors for common paths
# (monotonic, cyclic, nonproportional multiaxial, random walk). All builders
# produce UNIFORM time increments: the jax-fem kernels bake `problem.dt` in
# at the first JIT trace, so dt must be constant within one history (the
# driver asserts this).

from dataclasses import dataclass, field

import numpy as onp

SYM_SLOTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


@dataclass
class MacroStep:
    t: float                      # end time of the increment
    H_presc: onp.ndarray          # (3,3) prescribed component values
    mask: onp.ndarray             # (3,3) bool, True = prescribed
    sigma_targets: dict = field(default_factory=dict)  # {(i,j): MPa}

    def unknown_slots(self):
        """Ordered unknown slots from the symmetric set; validates the mask."""
        slots = []
        for (i, j) in SYM_SLOTS:
            if not self.mask[i, j]:
                if i != j:
                    assert not self.mask[j, i], \
                        f"unknown slot ({i},{j}) requires mask[{j},{i}]=False too"
                assert (i, j) in self.sigma_targets, \
                    f"unknown slot ({i},{j}) has no stress target"
                slots.append((i, j))
        # every mask=False entry must be covered by a symmetric slot
        n_free = int((~self.mask).sum())
        n_covered = sum(1 if i == j else 2 for (i, j) in slots)
        assert n_free == n_covered, \
            "mask has free components outside the symmetric slot set " \
            "(prescribe the skew part of H_bar explicitly)"
        return slots

    def assemble_H(self, theta):
        """Full (3,3) H_bar from prescribed values + unknown vector theta."""
        H = onp.array(self.H_presc, dtype=onp.float64)
        for a, (i, j) in enumerate(self.unknown_slots()):
            H[i, j] = theta[a]
            if i != j:
                H[j, i] = theta[a]
        return H


def _uniform_times(n_steps, t_total):
    return onp.linspace(0., t_total, n_steps + 1)[1:]


def _full_mask():
    return onp.ones((3, 3), dtype=bool)


def _uniaxial_stress_mask():
    """Prescribe H_zz and the skew part; free the five symmetric slots
    (xx, yy, xy, xz, yz) -- their targets are zero average stress."""
    mask = onp.ones((3, 3), dtype=bool)
    for (i, j) in [(0, 0), (1, 1), (0, 1), (0, 2), (1, 2)]:
        mask[i, j] = False
        mask[j, i] = False
    mask[2, 2] = True
    return mask


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def uniaxial_strain_path(eps_max=0.000, n_steps=2, t_total=0.000):
    """Fully prescribed F_bar = diag(1, 1, 1+eps(t)): the validation mode.
    Lateral macro strains held at zero (uniaxial-STRAIN state)."""
    steps = []
    for k, t in enumerate(_uniform_times(n_steps, t_total)):
        H = onp.zeros((3, 3))
        H[2, 2] = eps_max * (k + 1) / n_steps
        steps.append(MacroStep(t=t, H_presc=H, mask=_full_mask()))
    return steps


def uniaxial_stress_path(eps_max=0.01, n_steps=10, t_total=0.1):
    """Macroscopically uniaxial-STRESS tension: prescribe only H_zz = eps(t)
    (and the skew part = 0). H_xx and H_yy are INDEPENDENT unknowns, and the
    three symmetric shear slots are unknowns as well, all targeting zero
    volume-averaged stress:  <sigma_xx> = <sigma_yy> = <sigma_xy> =
    <sigma_xz> = <sigma_yz> = 0."""
    mask = _uniaxial_stress_mask()
    targets = {(0, 0): 0., (1, 1): 0., (0, 1): 0., (0, 2): 0., (1, 2): 0.}
    steps = []
    for k, t in enumerate(_uniform_times(n_steps, t_total)):
        H = onp.zeros((3, 3))
        H[2, 2] = eps_max * (k + 1) / n_steps
        steps.append(MacroStep(t=t, H_presc=H, mask=mask,
                               sigma_targets=dict(targets)))
    return steps


def cyclic_uniaxial_stress_path(eps_amp=0.005, n_cycles=2, n_per_quarter=25):
    """Triangular cyclic eps_zz(t): 0 -> +amp -> -amp -> ... under the same
    uniaxial-stress mixed control as uniaxial_stress_path."""
    mask = _uniaxial_stress_mask()
    targets = {(0, 0): 0., (1, 1): 0., (0, 1): 0., (0, 2): 0., (1, 2): 0.}
    # triangular wave sampled at uniform increments
    quarters = []
    for c in range(n_cycles):
        quarters += [(0., eps_amp), (eps_amp, -eps_amp), (-eps_amp, 0.)]
    # build a strictly uniform-increment sequence
    eps_vals = []
    for (e0, e1) in quarters:
        n = n_per_quarter * (2 if abs(e1 - e0) > 1.5 * eps_amp else 1)
        seg = onp.linspace(e0, e1, n + 1)[1:]
        eps_vals.extend(seg.tolist())
    dt = 0.001
    steps = []
    for k, eps in enumerate(eps_vals):
        H = onp.zeros((3, 3))
        H[2, 2] = eps
        steps.append(MacroStep(t=dt * (k + 1), H_presc=H, mask=mask,
                               sigma_targets=dict(targets)))
    return steps


def nonproportional_path(eps_max=0.005, gamma_max=0.005, n_steps_each=50):
    """Nonproportional multiaxial history: phase 1 ramps tension H_zz, phase 2
    holds H_zz and ramps a prescribed simple shear H_xz (nonsymmetric F_bar is
    allowed for prescribed components). Lateral response stays
    stress-controlled (<sigma_xx> = <sigma_yy> = <sigma_xy> = <sigma_yz> = 0);
    the sheared component (x,z) is prescribed, so it is removed from the
    unknown set."""
    mask = onp.ones((3, 3), dtype=bool)
    for (i, j) in [(0, 0), (1, 1), (0, 1), (1, 2)]:
        mask[i, j] = False
        mask[j, i] = False
    targets = {(0, 0): 0., (1, 1): 0., (0, 1): 0., (1, 2): 0.}
    dt = 0.001
    steps = []
    n = n_steps_each
    for k in range(2 * n):
        H = onp.zeros((3, 3))
        if k < n:                                    # tension ramp
            H[2, 2] = eps_max * (k + 1) / n
        else:                                        # hold tension, ramp shear
            H[2, 2] = eps_max
            H[0, 2] = gamma_max * (k + 1 - n) / n    # prescribed simple shear
        steps.append(MacroStep(t=dt * (k + 1), H_presc=H, mask=mask.copy(),
                               sigma_targets=dict(targets)))
    return steps


def random_walk_path(n_steps=50, dH_scale=1e-4, seed=0):
    """Fully prescribed random-walk history: symmetric random increments of
    H_bar (all nine components prescribed, skew part zero)."""
    rng = onp.random.default_rng(seed)
    dt = 0.001
    H = onp.zeros((3, 3))
    steps = []
    for k in range(n_steps):
        dH = rng.normal(0., dH_scale, size=(3, 3))
        dH = 0.5 * (dH + dH.T)          # symmetric increments
        H = H + dH
        steps.append(MacroStep(t=dt * (k + 1), H_presc=H.copy(),
                               mask=_full_mask()))
    return steps


# Presets build the FULL schedule and truncate to the first n steps when a
# step count is given (smoke tests keep the schedule's small increments --
# compressing the full strain into fewer steps would change the physics and
# risks diverging the stiff local Newton solve).
PRESETS = {
    'uniaxial-strain': lambda n=None: (uniaxial_strain_path()[:n]
                                       if n else uniaxial_strain_path()),
    'uniaxial-stress': lambda n=None: (uniaxial_stress_path()[:n]
                                       if n else uniaxial_stress_path()),
    'cyclic': lambda n=None: (cyclic_uniaxial_stress_path()[:n]
                              if n else cyclic_uniaxial_stress_path()),
    'nonproportional': lambda n=None: (nonproportional_path()[:n]
                                       if n else nonproportional_path()),
    'random-walk': lambda n=None: random_walk_path(n_steps=n or 50),
}
