#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:58:57 2022

@author: ender
"""

import numpy as np                           # For dealing with matrices in general
import matplotlib.pyplot as plt              # For drawing plots
import matplotlib.animation as animation     # For animating with plt
from cmath import exp                        # Complex exponential
from math import floor                       # Floor function

# Load variables
ψ = np.load("ψ.npy")
E = np.load("E.npy")
x = np.load("x.npy")
ħ = 0.0001

# Establish max time, time between eigenstate and fps
tmax = 60; tn = 10; fps = 30; dt = 1/fps; skip = 1
t = np.linspace(0, tmax, tmax*fps) # time vector
Ψ = np.zeros((ψ.shape[0],ψ.shape[1],t.size), dtype='complex')
# Calculation of the full wave function Ψ(x,t)
for i in range(0,tmax*fps):
    for j in range(0,ψ.shape[1]):
        Ψ[:,j,i] = ψ[:,j] * exp(-1j*E[j]*t[i]/ħ)

# Potential to plot
k = 0.25
def V(x):
    return k * (abs(x) > 1.5)

# Initialize Plt
fig = plt.figure()
ax = plt.axes()

# Function to update each frame
def update(i):
    print(i)
    ax.clear()                  # Clear
    ax.set_xlim([-30, 30])      # x limits
    ax.set_ylim([-0.3, 0.3])    # y limits
    l1, = ax.plot(x,V(x),'k')   # Potential
    l2, = ax.plot(x,np.absolute(Ψ[:,floor(i/(tn*fps*skip)),i]),'b')     # Absolute value of Ψ
    l3, = ax.plot(x,np.real(Ψ[:,floor(i/(tn*fps*skip)),i]),'r')         # Real oscillating part of Ψ
    ax.set_xlabel('x'); ax.set_ylabel('Ψ')      # Labels
    ax.legend([l1,l2,l3],['V','|Ψ|','Re{Ψ}'])   # Legend
    ax.set_xticks([])   # Remove ticks in x axis
    ax.set_yticks([])   # Remove ticks in y axis

ani = animation.FuncAnimation(fig, update, fps*tmax ,interval=1000*dt, repeat=False)
# Command to run the animation of figure fig, with function update, with fps*tmax frames
#ani.save('Sch2.mp4', writer="ffmpeg", dpi=600) # Saving the animation
fig.show() # Show the plot