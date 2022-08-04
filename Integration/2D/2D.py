#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:27:30 2022

@author: ender
"""

import numpy as np
from math import sqrt,pi

def rk4(f, u0, t0, tf , n):
    t = np.linspace(t0, tf, n)
    u = np.array((n)*[u0], dtype=complex)
    h = t[1]-t[0]
    for i in range(0,n-1):
        print(t[i]*100/tf)
        k1 = h * f(u[i], t[i])
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)
        k4 = h * f(u[i] + k3, t[i] + h)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6
    return u, t

def laplacian2D(z,x,y):
    hx = x[0,1]-x[0,0]
    hy = y[1,0]-y[0,0]
    nx = x.shape[1]
    ny = y.shape[0]
    L = np.zeros_like(z)
    L[0,:] = np.zeros(nx)
    L[ny-1,:] = np.zeros(nx)
    L[:,0] = np.zeros(ny)
    L[:,nx-1] = np.zeros(ny)
    for j in range(1,nx-1):
        L[1,j] += (z[2,j]+z[0,j]-2*z[1,j])/(hy**2)
        L[ny-2,j] += (z[ny-1,j]+z[ny-3,j]-2*z[ny-2,j])/(hy**2)
    for i in range(1,ny-1):
        L[i,1] += (z[i,2]+z[i,0]-2*z[i,1])/(hx**2)
        L[i,nx-2] += (z[i,nx-1]+z[i,nx-3]-2*z[i,nx-2])/(hx**2)
    for i in range(1,ny-1):
        for j in range(2,nx-2):
            L[i,j] += (-z[i,j+2]+16*z[i,j+1]-30*z[i,j]+16*z[i,j-1]-z[i,j-2])/(12*hx**2)
    for i in range(2,ny-2):
        for j in range(1,nx-1):
            L[i,j] += (-z[i+2,j]+16*z[i+1,j]-30*z[i,j]+16*z[i-1,j]-z[i-2,j])/(12*hy**2)
    return L

#hbar = 1.05457182e-34
hbar = 1
def Sch(Psi,t,x,y,m,V):
    #P = V(x,y,t) 
    T = (hbar**2/(2*m)) * laplacian2D(Psi,x,y)
    dPsi = np.zeros_like(Psi)
    dPsi = (T-P*Psi)*1j/hbar
    return dPsi

# k = 0.1
# def V(x,y,t):
#     return (k/2)*(x**2+y**2)

# k = 0.08
# def V(x,y,t):
#     return (k/2)*((k*x**2 + k*y**2 -3)**2)

k = 1
def V(x,y,t):
    return -k/np.sqrt(x**2+y**2)

def V0(x,y,t):
    return np.zeros_like(x)

ax = -3; ay = -3; nx = 200; ny = nx; sigmax = 1; sigmay = sigmax; vx = 10; vy = 10
x = np.linspace(-10,10,nx, dtype=complex)
y = np.linspace(-10,10,ny, dtype=complex)
xv, yv = np.meshgrid(x,y)
Psi0 = sqrt(1/(sigmax * sigmay * sqrt(2*pi)))*np.exp(-((xv-ax)/sigmax)**2/4 -((yv-ay)/sigmay)**2/4 )*np.exp(-1j*vx*xv -1j*vy*yv)
nx = xv.shape[1]
ny = yv.shape[0]
Psi0[0,:] = np.zeros(nx)
Psi0[ny-1,:] = np.zeros(nx)
Psi0[:,0] = np.zeros(ny)
Psi0[:,nx-1] = np.zeros(ny)

P = V0(xv,yv,0)
m=10; fps = 30; tmax = 90; skip = 1; nt = int(fps*tmax*skip)
t = np.linspace(0, tmax, nt); dt = t[1]-t[0]; dx = np.real(x[1]-x[0])
print(hbar*dt/(2*m*dx**2))
Psi, t = rk4(lambda Psi,t: Sch(Psi,t,xv,yv,m,P),Psi0,0,tmax,nt)
np.save("Psi.npy",Psi)
np.save("t.npy",t)
np.save("x.npy",xv)
np.save("y.npy",yv)