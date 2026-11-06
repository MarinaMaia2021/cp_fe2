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

INPUT:  
OUTPUT:  

# Environment 
environment.yml

# Validation 
To be done: 

verify _single_point_gp.py_ gives the same pairs of homogenized quantities as using _problem.py_
