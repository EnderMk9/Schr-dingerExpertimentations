#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:58:57 2022

@author: ender
"""

import numpy as np      # For dealing with matrices in general
from mayavi import mlab # Library to plot in 3D

# Load the computen eigenstates and the rest of the neded variables
ψ = np.load("ψ.npy")
E = np.load("E.npy")

ax_ranges = [-10, 10, -10, 10, -10, 10];
# Range of the box

# Function to save a photo, with options to chose the eigenstate, opacity, suffix for the file and to close or not the figure to change pov
def photo(i,op=0.9,suf='',close=True):
    a = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    s1 = mlab.contour3d(ψ[:,:,:,i],colormap='jet', extent=ax_ranges,contours=8,opacity=op)
    mlab.outline(s1,color=(0,0,0),opacity=0.5)
    mlab.savefig('./frame/H' + str(i) + suf +'.png')
    if close == True:
        mlab.close()

# loop to generate a range of photos of the eigenstates
for i in range(0,E.size):
    photo(i,1,'op')
    photo(i,0.9,'tr')