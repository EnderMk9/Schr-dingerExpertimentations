#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:27:30 2022

@author: ender
"""

#from cmath import *
import numpy as np
from math import sqrt,pi
#from pylab import plot,show  # Matlab-like plots; show() to show
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rk4(f, u0, t0, tf , n):
    t = np.linspace(t0, tf, n)
    u = np.array((n)*[u0], dtype=complex)
    h = t[1]-t[0]
    for i in range(0,n-1):
        k1 = h * f(u[i], t[i])
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)
        k4 = h * f(u[i] + k3, t[i] + h)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6
    return u, t

def derv2(y,x):
    h = x[1]-x[0]
    dP2 = np.zeros_like(y)
    #dP2[0] = (2*y[0]-5*y[1]+4*y[2]-y[3])/(h**3)
    dP2[0] = 0
    dP2[1] = (y[2]+y[0]-2*y[1])/(h**2)
    for i in range(2,y.size-2):
        dP2[i] = (-y[i+2]+16*y[i+1]-30*y[i]+16*y[i-1]-y[i-2])/(12*h**2)
    dP2[y.size-2] = (y[y.size-1]+y[y.size-3]-2*y[y.size-2])/(h**2)
    #dP2[y.size-1] = (2*y[y.size-1] -5*y[y.size-2]+4*y[y.size-3]-y[y.size-4])/(h**3)
    dP2[y.size-1] = 0
    return dP2

#hbar = 1.05457182e-34
hbar = 1
def Sch(Psi,t,x,m,V):
    P = V(x,t) 
    T = (hbar**2/(2*m)) * derv2(Psi,x)
    dPsi = np.zeros_like(Psi)
    for i in range(0,Psi.size):
        dPsi[i] = (T[i]-P[i]*Psi[i])*1j/hbar
    return dPsi

k = 2.2
def V(x,t):
    return k*((x > 1) * (x < 2))

def V0(x,t):
    return np.zeros_like(x)

a = -4; nx = 1000; sigma = 0.8;
x = np.linspace(-10,10,nx, dtype=complex)
Psi0 = sqrt(1/(sigma * sqrt(2*pi)))*np.exp(-((x-a)/sigma)**2/4)*np.exp(-1j*(-25)*x)

m=150; fps = 30; tmax = 180; skip = 2; nt = int(fps*tmax*skip)
t = np.linspace(0, tmax, nt); dt = t[1]-t[0]; dx = np.real(x[1]-x[0])
print(hbar*dt/(2*m*dx**2))
Psi, t = rk4(lambda Psi,t: Sch(Psi,t,x,m,V),Psi0,0,tmax,nt)
np.save("Psi.npy",Psi)
np.save("t.npy",t)
np.save("x.npy",x)
