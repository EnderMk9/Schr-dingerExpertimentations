#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:58:57 2022

@author: ender
"""

import numpy as np                         # For dealing with matrices in general
from math import floor                     # Floor function
from scipy.sparse.linalg import eigs       # Very efficient Diagonalization function
from scipy.sparse import diags             # Function to create sparse matrix from diagonals

En = 24     # Number of energies to obtain

# Solves eigenvalue equation ∇²ψ  + r(x,y,z)ψ = λ ψ, for boundary conditions ψ = 0 at the borders
# Inputs:
#       r (function of three variables)
#       h (float): Δx = Δy = Δz
#       (a0,af) (floats): (x0,xf) = (y0,yf) = (z0,zf)
# Returns:
#       λ (En,) array              : Eigenvalues
#       χ (N+2,N+2,N+2,En) array   : Eigenvectors
#       x,y,z (N+2,N+2,N+2) arrays : meshes

def FiniteDiffDiag(r,h,a0,af):
    N = int((af-a0)/h)-2        # Number of intervals excluding the borders
    x = np.linspace(a0,af,N+2)  # x array, including the borders
    y = np.linspace(a0,af,N+2)  # y array, including the borders
    z = np.linspace(a0,af,N+2)  # z array, including the borders
    # If you want to really understand this, you should write ∇²ψ  + r(x,y,z)ψ = λ ψ, discretize using finite differences for the laplacian,
    # the laplacian, then unravel the vector ψ and compose the matrix so you recover the set of equations when you operate on the vector ψ.
    a = np.zeros(N**3); b = np.ones(N**3-1); c = np.ones(N**3-N); d = np.ones(N**3-N**2)
    # a (N³,) array    : 0th diagonal of the sparse matriz, excluding borders. This corresponds with the terms ψ_ijk of the discretized diff. eq.
    # b (N³-1,) array  : 1st and -1st diagonals, adjacent to a, of the sparse matrix, excluding borders. These correspond with the terms ψ_i±1 jk
    # c (N³-N,) array  : Nth and -Nth diagonals of the sparse matrix, excluding borders, These correspond with the terms ψ_i j±1 k
    # d (N³-N²,) array : N²th and -N²th diagonals of the sparse matrix, excluding borders, These correspond with the terms ψ_i j k±1
    # b, c and d are mostly full of ones.
    # The matrix s we're gonna build acts upon an unraveled 3D array, that means that ψ_ijk gets turned to ψ_i, with i running from 0 to
    # N³-1, each group of N elements is a row of ψ_(i)jk, each group of N² contains the 2D arrays ψ_(ij)k.
    for i in range(0,N**3-1):
        # For each row, we must set this elements to 0 so that the beggining of a new row does not interact directly with end of the
        # previous one, and viceversa, otherwise we would encounter looped patterns in the edges.
        if ((i+1)% N) == 0:
            b[i] = 0
    for i in range(0,N**3-N):
        # The same, but in this case, to prevent a whole row in the beggining of one of the 2D arrays to directly interact with the last row of
        # the previous 2D array, and viceversa.
        if ((i+N)% (N**2)) < N:
            c[i] = 0
    for i in range(0,N**3):
        # These are the values of the 0th diagonal, we have to loop the coordinates to de-unravel the index i
        a[i] = h*h*r(x[(i)%(N)+1],y[(floor((i)/(N))+1)%N +1],z[floor((i)/(N**2))+1])-6
    s = diags([a,b,b,c,c,d,d],[0,1,-1,N,-N,N**2,-N**2])     # This function builds the sparse matrix out of the diagonals
    λ,ξ = eigs(s, which='LR', k = En)                       # This function computes the En eigenvalues with largest real part and their eigenvectors
    χ = np.zeros((x.size,y.size,z.size,En))                 # (N+2,N+2,N+2,En) in which we are going to de-unravel the eigenvectors, includes borders
    χ[1:x.size-1,1:y.size-1,1:z.size-1,:] = np.real(np.reshape(ξ,(x.size-2,y.size-2,z.size-2,En),order='F')) # de-unravel
    xv, yv, zv = np.meshgrid(x,y,z)                         # Meshes to return and then plot
    return np.real(λ),χ,x,y,z


k = 8
def V(x,y,z): # k/r potencial, Hydrogen Atom
    return -k/np.sqrt(x**2+y**2+z**2)

ħ = 1               # Planks reduced constant
m = 1               # Mass of the particle
cte = -ħ**2/(2*m)   # Constant that multiplies the Laplacian in Schrödinger's equation
# Solve for ∇²ψ  + (V/cte)ψ = (E/cte) ψ
λ,ψ,x,y,z = FiniteDiffDiag(lambda x,y,z: V(x,y,z)/cte, 0.125, -10, 10)
# h = 0.2 - 10 min | h = 0.125 - 1 hora | h = 0.1 - crash

E = λ*cte
# The energies are sorted from lowest to highest (thanks to cte < 0)

np.save("ψ.npy",ψ); np.save("E.npy",E)
np.save("x.npy",x); np.save("y.npy",y); np.save("z.npy",z)
# Save the solutions for later use
