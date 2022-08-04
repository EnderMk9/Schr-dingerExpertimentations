#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:58:57 2022

@author: ender
"""

import numpy as np                         # For dealing with matrices in general
from scipy.sparse.linalg import eigsh      # Very efficient Diagonalization function
from scipy.sparse import diags             # Function to create sparse matrix from diagonals

# solves eigenvalue equation y'' + p(x)y' + q(x)y= λ y, for boundary conditions y(x0) = 0 and y(xf) = 0
# Inputs:
#       p,q (function of one variable)
#       N (int): Number of intervals excluding borders
#       (x0,xf) (floats)
# Returns:
#       λ (En,) array      : Eigenvalues
#       ξ (N+2,En) array   : Eigenvectors
#       x (N+2) array      : x vector
def FiniteDiffDiag(p,q,N,x0,xf):
    h = (xf-x0)/(N+1)           # separation between intervals
    x = np.linspace(x0,xf,N+2)  # x vector, including borders
    # Diagonals for finite difference matrix
    a = np.zeros(N); b = np.zeros(N-1); c = np.zeros(N-1);
    for i in range(0,N-1):
        b[i] = 1-h*p(x[i+1])/2.
    for i in range(0,N):
        a[i] = h*h*q(x[i+1])-2
    for i in range(1,N):
        c[i-1] = 1+h*p(x[i+1])/2.
    s = diags([b,a,c],[-1,0,1]) # Finite difference sparse matrix made from diagonals
    λ,ξ = eigsh(s,k=6,which='SM') # This function computes the En eigenvalues with smallest magnitude and their eigenvectors
    return λ,ξ,x

# Harmonic Potential
# k = 10 #[-10 10]
# def V(x):
#     return (k/2)*x**2

# Finite Box Potential
k = 0.5
def V(x):
    return k * (abs(x) > 5)

ħ = 1               # Planks reduced constant
m = 1               # Mass of the particle
cte = -ħ**2/(2*m)   # Constant that multiplies the second derivative in Schrödinger's equation

# Solve for ψ''  + (V/cte)ψ = (E/cte) ψ
λ,ξ,x = FiniteDiffDiag(lambda x: 0, lambda x: V(x)/cte, 2000, -100, 100)

E = λ*cte
# Flipping the order so that its sorted from lowest to highest energy
E = np.flip(E)
ψ = np.flip(ξ,1)
# Including the borders in ψ
ψ = np.insert(ψ,0,np.zeros(ψ.shape[1]),axis=0)
ψ = np.insert(ψ,ψ.shape[0],np.zeros(ψ.shape[1]),axis=0)

# Save the solutions for later use
np.save("ψ.npy",ψ); np.save("E.npy",E); np.save("x.npy",x)
