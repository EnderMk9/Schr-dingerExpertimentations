#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:58:57 2022

@author: ender
"""

import numpy as np                         # For dealing with matrices in general
from math import floor                     # Floor function
from scipy.sparse.linalg import eigsh      # Very efficient Diagonalization function
from scipy.sparse import diags             # Function to create sparse matrix from diagonals

En = 12     # Number of energies to obtain

# solves eigenvalue equation ∇²ψ + r(x,y)ψ = λ ψ, for boundary conditions ψ = 0 at the borders
# Inputs:
#       r (function of two variables)
#       h (float): Δx = Δy 
#       (a0,af) (floats): (x0,xf) = (y0,yf) 
# Returns:
#       λ (En,) array              : Eigenvalues
#       χ (N+2,N+2,En) array   : Eigenvectors
#       x,y,z (N+2,N+2) arrays : meshes
def FiniteDiffDiag(r,h,a0,af):
    N = int((af-a0)/h) - 2     # Number of intervals excluding the borders
    x = np.linspace(a0,af,N+2) # x array, including the borders
    y = np.linspace(a0,af,N+2) # y array, including the borders
    # If you want to really understand this, you should write ∇²ψ  + r(x,z)ψ = λ ψ, discretize using finite differences for the laplacian,
    # the laplacian, then unravel the vector ψ and compose the matrix so you recover the set of equations when you operate on the vector ψ.
    a = np.zeros(N**2); b = np.ones(N**2-1); c= np.ones(N**2-N);
    # a (N²,) array    : 0th diagonal of the sparse matriz, excluding borders. This corresponds with the terms ψ_ij of the discretized diff. eq.
    # b (N²-1,) array  : 1st and -1st diagonals, adjacent to a, of the sparse matrix, excluding borders. These correspond with the terms ψ_i±1 j
    # c (N²-N,) array  : Nth and -Nth diagonals of the sparse matrix, excluding borders, These correspond with the terms ψ_i j±1 
    # b and c are mostly full of ones.
    # The matrix s we're gonna build acts upon an unraveled 2D array, that means that ψ_ij2 gets turned to ψ_i, with i running from 0 to
    # N²-1, each group of N elements is a row of ψ_(i)j
    for i in range(0,N**2-1):
        # For each row, we must set this elements to 0 so that the beggining of a new row does not interact directly with end of the
        # previous one, and viceversa, otherwise we would encounter looped patterns in the edges.
        if ((i+1)% N) == 0:
            b[i] = 0
    for i in range(0,N**2):
        # These are the values of the 0th diagonal, we have to loop the coordinates to de-unravel the index i
        a[i] = h*h*r(x[(i)%(N)+1],y[floor((i)/(N))+1])-4
    s = diags([a,b,b,c,c],[0,1,-1,N,-N])                # This function builds the sparse matrix out of the diagonals
    λ,ξ = eigsh(s, which='SM', k = En)                  # This function computes the En eigenvalues with samllest magnitude and their eigenvectors
    χ = np.zeros((x.size,y.size,En))                    # (N+2,N+2,En) in which we are going to de-unravel the eigenvectors, includes borders
    χ[1:x.size-1,1:y.size-1,:] = ξ = np.reshape(ξ,(x.size-2,y.size-2,En),order='F') # de-unravel
    xv, yv = np.meshgrid(x,y) # grids
    return λ,χ,xv,yv

# Harmonic Potentia
k = 10
def V(x):
    return (k/2)*(x**2+y**2)

# Hat Potential
# k = 1
# def V(x,y):
#     return (k/2)*((k*x**2 + k*y**2 -3)**2)


ħ = 1               # Plank's reduced constant
m = 1               # Mass of the particle
cte = -ħ**2/(2*m)   # Constant that multiplies the Laplacian in Schrödinger's equation
# Solve for ∇²ψ  + (V/cte)ψ = (E/cte) ψ
λ,ψ,x,y = FiniteDiffDiag(lambda x,y: V(x,y)/cte, 0.1, -10, 10,-10,10)

E = λ*cte
# Flipping the order so that its sorted from lowest to highest energy
E = np.flip(E)
ψ = np.flip(ψ,2)

# Save the solutions for later use
np.save("ψ.npy",ψ); np.save("E.npy",E)
np.save("x.npy",x);np.save("y.npy",y)
