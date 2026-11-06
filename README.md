# Multiscale 
Run multiscale simulation using the RVE described below. 
At the moment, the macroscopic domain consists of a 3D mesh made of TET4s, with one (to be checked) single integration point associated to each element. 

File name: _fe2_cp.py_ 

INPUT:  
OUTPUT:  

# Microscale (RVE) 
Micromodel generated with 8 grains and orientations defined in .msh and .tess. TET4s elements are adopted.
Constitutive model: _models_304steel.py_
It was rewritten to be called more easily in the file _rve_crystal_plasticity.py_ 

File name: _single_point_gp.py_ 

INPUT: Homogenized deformation gradient, increments of time per step
OUTPUT: Homogenized cauchy stresses, Homogenized tangent stiffness matrix

# Environment 
Version of packages used to run this repo
environment.yml

# To-do list: 
  1. Verify _single_point_gp.py_ gives the same pairs of homogenized quantities as using _problem.py_
  2. Generate samples with unrotated pairs of homogenized stretches and unrotated homogenized (cauchy) stresses
  3. Create PRNN file with embedded crystal plasticity model (with simple linear encoder and decoder)
