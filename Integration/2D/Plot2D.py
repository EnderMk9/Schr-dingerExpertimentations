#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:12:41 2022

@author: ender
"""

import numpy as np
from mayavi import mlab

k = 10
def V(x,y,t):
    return -k/np.sqrt(x**2+y**2) + 2

fps = 30; tmax = 45; skip = 6; nt = int(fps*tmax*skip)

print('LOADING')
Psi = np.flip(np.load("Psi.npy"),(1,2))
print('LOADED')
t = np.load("t.npy")
x = np.transpose(np.load("x.npy"))
y = np.transpose(np.load("y.npy"))


ax_ranges = [-10, 10, -10, 10, 0, 3]; i = 0
mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.title('t = ' + "{:.2f}".format(t[skip*i]) + ' s', size=0.25, height=0.85)
#l1 = mlab.plot3d(np.real(x[:,0]),np.real(y[:,int(y.shape[0]/2)]),np.real(V(x[:,0],y[:,int(y.shape[0]/2)],t[i])),color=(0,0,0), extent=(-10,10,0,0,0,3),tube_radius=0.15)
#l2 = mlab.plot3d(np.real(x[int(x.shape[0]/2),:]),np.real(y[0,:]),np.real(V(x[int(x.shape[0]/2),:],y[0,:],t[i])),color=(0,0,0), extent=(0,0,-10,10,0,3),tube_radius=0.15)
s1 = mlab.surf(np.absolute(Psi[i*skip,:,:]),colormap='jet', extent=ax_ranges)
mlab.savefig('/mnt/data/frame/f' + str(i).zfill(4) + '.png')

for i in range(1,fps*tmax):
     print(i)
     mlab.title('t = ' + "{:.2f}".format(t[skip*i]) + ' s', size=0.25, height=0.85)
     s1.mlab_source.scalars = np.absolute(Psi[i*skip,:,:])
     mlab.savefig('/mnt/data/frame/f' + str(i).zfill(4) + '.png')