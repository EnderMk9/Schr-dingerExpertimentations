#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:58:57 2022

@author: ender
"""

import numpy as np         # For dealing with matrices in general
from mayavi import mlab    # Library to plot in 3D
from math import floor,cos # cos and floor libraries

# Load the computen eigenstates and the rest of the neded variables
ψ = np.load("ψ.npy")
E = np.load("E.npy")
x = np.load("x.npy").transpose()
y = np.load("y.npy").transpose()
# x and y are transposed because mayavi reads them differently than numpy and plt
ħ = 0.01 # Plank's reduced constant

# Establish max time, time between eigenstate and fps
tmax = 1800; tn = 5; fps = 1; dt = 1/fps; skip = 30
t = np.linspace(0, tmax, tmax*fps)  # time vector

# Harmonic Potential
# k = 0.25
# def V(x,y):
#     return (k/2)*(x**2+y**2)

# Hat potential
# k = 0.1
# def V(x,y):
#     return (k/2)*((k*x**2 + k*y**2 -3)**2)

Idf = 0 # To adjust the height of the potential lines
ax_ranges = [-10, 10, -10, 10, 0, 5]; i = 0
# extent of the figure and 0th iteration
mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# colors for the figure
#l1 = mlab.plot3d(np.real(x[:,0]),np.real(y[:,int(y.shape[0]/2)]),df+np.real(V(x[:,0],y[:,int(y.shape[0]/2)])),extent=(-3,3,0,0,df,df+5),color=(0,0,0),tube_radius=0.15, opacity=0.5)
#l2 = mlab.plot3d(np.real(x[int(x.shape[0]/2),:]),np.real(y[0,:]),df+np.real(V(x[int(x.shape[0]/2),:],y[0,:])),extent=(0,0,-3,3,df,df+5),color=(0,0,0),tube_radius=0.15, opacity=0.5)
# Potential lines
s1 = mlab.surf(ψ[:,:,floor(i/(tn*fps*skip))],colormap='jet', extent=ax_ranges)
# Surface of the wave function

# Loop to animate the wave function
for i in range(0,fps*tmax):
     print(i)
     # update the wave function multiplying by the real part of exp(-1j*E[j]*t[i]/ħ) and changing eigenstate after tn time
     s1.mlab_source.scalars = ψ[:,:,floor(i/(tn*fps*skip))]*cos(E[floor(i/(tn*fps*skip))]*t[i]/ħ)
     mlab.savefig('./frame/f' + str(i).zfill(4) + '.png')
     # save frame