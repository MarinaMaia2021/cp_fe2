gp_path_seed_000_voronoi_8grain_cube_cut -- deformation gradient and Cauchy stress
of one random (GP) strain path on the coarse 8-grain Voronoi RVE, for comparison
with your code

RVE
Exactly the files sent on 2026-10-02 (cp_rve_box_8grain_coarse):
rve8_box_periodicity.msh / rve8_box.msh (522 nodes, 2,132 TET4, unit cube) and
rve8.tess (orientations). Checked byte for byte against the mesh used here.

Files
gp_path_seed_000.txt              the input strain path: 60 rows (GP step 0..59),
                                  eps_xx eps_yy eps_zz eps_xy eps_xz eps_yz,
                                  logarithmic strain, tensor shear (not engineering)
..._at_gp_steps.csv               results at the 60 GP steps (one row per path row)
..._every_increment.csv           results at all 295 solver increments
                                  (5 increments per GP step) plus the initial state
gp_path_seed_000_voronoi_8grain_cube_cut.png/.svg   figure of both tables
material_parameters.json          crystal plasticity parameters used

CSV columns
increment, gp_step, t_s (time in s),
F_xx F_xy F_xz F_yx F_yy F_yz F_zx F_zy F_zz  (macroscopic deformation gradient,
                                               F_ij = dx_i/dX_j),
sigma_xx sigma_yy sigma_zz sigma_xy sigma_xz sigma_yz  (volume-averaged Cauchy
                                               stress, MPa)

Loading
- Strain control: all nine components of F are prescribed at every increment,
  F = exp(eps) with eps the log strain of the path file. F is therefore symmetric
  (pure stretch, no rigid rotation): F_xy = F_yx etc.
- Between two GP steps eps is interpolated linearly in 5 increments.
- One GP step lasts 0.01 s (one increment 0.002 s), about 0.1 /s strain rate.
  The model is rate dependent, so please use the same times.
- Path: GP with RBF kernel (variance 8e-4, length scale 20 steps, 60 steps) on the
  5 deviatoric components plus a volumetric GP bounded at +-0.2 % ("option A",
  your generator with a volumetric limit). Peak strains about 3 %.

Boundary conditions and solution
- Periodic: u = (F - I) X + w, w periodic on the paired nodes of opposite faces,
  one node pinned. Stress = volume average of the element Cauchy stresses.
- JAX-CPFEM Kalidindi model (FCC, 12 {111}<110> systems), TET4 with one
  quadrature point, implicit time integration, global Newton per increment.
  Converged with one FE solve per increment.
- Material: the published JAX-CPFEM SS304 parameters (not the Core 304
  calibration): g_ini 90 MPa, h_0 392.9772 MPa, g_sat 7295.1754 MPa, a = 8,
  gamma_dot_0 1e-3 /s, m = 1/120, latent hardening ratio 1; elastic constants
  C11 = 262.2, C12 = 112.0, C44 = 74.6 GPa.
- Run on 2026-10-12 (GPU, double precision).

Accuracy notes for the comparison
- Time discretisation: on the cubic 8-grain RVE, 5 increments per GP step differ
  by about 0.75 % from 10 increments per GP step. If your increments differ from
  ours, differences of this order are expected.
- Compare at the GP steps (the _at_gp_steps file) if your code uses another
  number of increments.

Final state (GP step 59)
sigma_xx 90.5, sigma_yy -206.6, sigma_zz 44.7, sigma_xy 117.2, sigma_xz -25.6,
sigma_yz 272.7 MPa
