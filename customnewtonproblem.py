#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:53:41 2026

@author: malvesmaia
"""
from auxFunctions import _setup_solver

class CustomNewtonProblem:
    def __init__(self, quadrature_map, 
                 F, J, u, bcs,
                 verbose = False,
                 directSolver = False,
                 max_it=30, rtol=1e-5, atol=1e-6, max_growth_its=3):
        from dolfinx.fem import form, Function
        from dolfinx.fem.petsc import create_matrix, create_vector
        
        # Integration quadrature
        self.qmap = quadrature_map
        self.verbose = verbose
        self.macroSolverType = directSolver
        
        # Safe form compilation checking
        if isinstance(F, list):
            self.L = [form(f) if not hasattr(f, "function_space") else f for f in F]
        else:
            self.L = form(F) if not hasattr(F, "function_space") else F
            
        if isinstance(J, list):
            self.a = [form(j) if not hasattr(j, "function_space") else j for j in J]
        else:
            self.a = form(J) if not hasattr(J, "function_space") else J
            
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u
        self.du = Function(self.u.function_space, name="Increment")
        self.it = 0

        # Matrix preallocation via modern form objects
        if isinstance(self.a, list):
            self.A = create_matrix(self.a[0])
            self.b = create_vector(self.L[0].function_spaces)
        else:
            self.A = create_matrix(self.a)  
            self.b = create_vector(self.L.function_spaces)
        
        self.max_it = max_it
        self.max_growth_its = max_growth_its    # consecutive iterations with growing residual
        self.rtol = rtol
        self.atol = atol
        self.internal_forces = None      # for retrieving internal forces easily
        self.comm = self.u.function_space.mesh.comm
        self.solver = _setup_solver(self.qmap.macro_domain, directSolver=self.macroSolverType, tag='macro_')
        
        if self.verbose: print('Finished initiating Newton Raphson.')
                
    # Main solve function
    def solve(self):
        from dolfinx.common import Timer
        import petsc4py.PETSc as PETSc
        from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
                  
        # Handling different version names                   
        if hasattr(self.u.x, "petsc_vec"):
            u_petsc = self.u.x.petsc_vec
        else:
            u_petsc = self.u.x.petsc_vector

        if hasattr(self.du.x, "petsc_vec"):
            du_petsc = self.du.x.petsc_vec
        else:
            du_petsc = self.du.x.petsc_vector
        
        it = 0  # number of iterations of the Newton solver
        allMicroConverged = False
        macro_converged = False
        growth_its = 0
        is_root = self.comm.rank == 0
        
        while it < self.max_it:            
            with Timer("Constitutive update"):
                if self.verbose: print('Update integration points.')
                allMicroConverged = self.qmap.update(self.u)
                if self.verbose: print('End of update integration points')
            
            # Stop timestep if a micromodel does not not converge
            if not allMicroConverged:
                if is_root: print("  [FAILURE] One or more micromodels did not converge.", flush=True)
                return False, it    # TODO: implement substepping for specific micromodel
                      
            # Assemble Jacobian and residual
            with self.b.localForm() as loc_b:
                loc_b.set(0)
                
            if isinstance(self.L, list):
                 for Li in self.L:
                     bi = assemble_vector(Li)
                     self.b.axpy(1.0, bi)
            else:
                 assemble_vector(self.b, self.L)
                 
            if self.verbose:
                raw_b = self.b.getArray(readonly=True).copy()
                print(f"\n[Macro] [Iter {it}] Residual vector (pre-lifting):", flush=True)
                print(raw_b, flush = True)
            
            # Compute b - alpha * J(u_D-u_(i-1))
            if isinstance(self.a, list):
                for ai in self.a:
                    apply_lifting(self.b, [ai], [self.bcs], x0=[u_petsc], alpha=-1.0)
            else:
                apply_lifting(self.b, [self.a], [self.bcs], x0=[u_petsc], alpha=-1.0)
                              
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )       
            
            if self.verbose:
                raw_b_lifted = self.b.getArray(readonly=True).copy()
                print(f"\n[Macro] [Iter {it}] Raw residual (lifted):", flush=True)
                print(raw_b_lifted, flush = True)
            
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(self.b, self.bcs, u_petsc, -1.0)        
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )
            
            if self.verbose:
                raw_b_lifted_bc = self.b.getArray(readonly=True).copy()
                print(f"\n[Macro] [Iter {it}] Raw residual (lifted+bc):", flush=True)
                print(raw_b_lifted_bc, flush = True)

            # Calculate norm of residual AFTER boundary conditions are applied
            norm_res = self.b.norm(PETSc.NormType.NORM_2)
            # The first residual holds the jump of the prescribed displacements, so
            # the reference is the first residual with the boundary conditions satisfied
            if it <= 1:
                norm_ref = norm_res
            rel_norm_res = norm_res / norm_ref if norm_ref > 0.0 else 0.0
    
            if is_root:
                print(f"[Macro] [Iter {it:2d}] | Residual norm: {norm_res:.4e}"
                      f" Relative residual norm: {rel_norm_res:.4e}", flush=True)
    
            # Near a kink of the response (e.g. reverse yielding) the residual can grow
            # for one iteration and still converge, so only sustained growth is divergence
            growth_its = growth_its + 1 if it > 0 and norm_res > 1.5*prev_res else 0
            if growth_its >= self.max_growth_its:
                # Divergence. Try with smaller step.
                if is_root:
                    print(f"[Macro] [DIVERGENCE] Residual norm: {norm_res:.4e}"
                          f" Previous residual norm: {prev_res:.4e}", flush=True)
                return False, it

            if (it > 0 and rel_norm_res < self.rtol) or norm_res < self.atol:
                macro_converged = True
                break

            # Assemble tangent matrix with BCs
            self.A.zeroEntries()
            if isinstance(self.a, list):
                self.A.assemble()
                for ai in self.a:
                    Ai = assemble_matrix(ai, bcs=self.bcs)
                    Ai.assemble()
                    self.A.axpy(1.0, Ai)
            else:
                assemble_matrix(self.A, self.a, bcs=self.bcs)
                self.A.assemble()

            # Compute negative residual vector
            self.b.scale(-1)

            # Solve linear problem K * du = -R
            self.solver.setOperators(self.A)
            
            with Timer("Linear solve"):
                try:
                    self.solver.solve(self.b, du_petsc)
                except PETSc.Error:
                    # Singular tangent: the caller retries with a smaller step. A failed
                    # factorization leaves the solver unusable until it is reset
                    self.solver.reset()
                    if is_root: print("[Macro] [FAILURE] Linear solve failed.", flush=True)
                    return False, it
                self.du.x.scatter_forward()

            # Update displacement candidate: u_{k+1} = u_k + du
            self.u.x.array[:] += self.du.x.array[:]
            self.u.x.scatter_forward()
            
            if self.verbose:
                print('[Macro] Increment of displacement ', self.du.x.array, flush = True)
                print('[Macro] Total displacement ', self.u.x.array, flush = True)
            
            it += 1
            prev_res = norm_res
                    
        return macro_converged, it
