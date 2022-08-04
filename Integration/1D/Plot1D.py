#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:12:41 2022

@author: ender
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

k = 2
def V(x,t):
    return k*((x > 1) * (x < 2))

fps = 30; tmax = 10; skip = 10; nt = int(fps*tmax*skip)

Psi = np.load("Psi.npy")
t = np.load("t.npy")
x = np.load("x.npy")

fig = plt.figure()
ax = plt.axes()

def update(i):
    print(i)
    ax.clear()
    ax.set_ylim([-5, 10])
    l1, = ax.plot(x,V(x,t[skip*i]),'k')
    l2, = ax.plot(x,np.absolute(Psi[skip*i,:]),'b')
    l3, = ax.plot(x,np.real(Psi[skip*i,:]),'r')
    ax.set_xlabel('x'); ax.set_ylabel('Ψ')
    ax.legend([l1,l2,l3],['V','|Ψ|','Re{Ψ}'])
    ax.set_title('t = ' + "{:.2f}".format(t[skip*i]) + ' s')
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, fps*180 ,interval=1000/fps, repeat=False)
ani.save('Sch5.mp4', writer="ffmpeg", dpi=600)
#fig.show()