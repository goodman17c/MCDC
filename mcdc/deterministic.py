import math

from mpi4py import MPI
from numba  import njit, objmode, literal_unroll

import matplotlib.pyplot as plt

import numpy as np


#Quasidiffusion
def QD1D(mcdc, phi_in, J_in, Edd):
    t = mcdc['technique']['ww_mesh']['t']
    x = mcdc['technique']['ww_mesh']['x']
    dx = x[1:]-x[:-1]
    Nx = len(dx)
    print(Nx)
    dx2 = np.zeros([1,Nx+1])
    dx2[0] = dx[0]/2
    dx2[1:-1] = (dx[1:]+dx[:-1])/2
    dx2[-1] = dx[-1]/2
    print(dx2)
    dt = t[mcdc['technique']['census_idx']+1]-t[mcdc['technique']['census_idx']]
    sigmaT = mcdc['materials'][0]['total'][0]
    sigmaA = mcdc['materials'][0]['capture'][0]
    sigmaF = mcdc['materials'][0]['fission'][0]
    nu = mcdc['materials'][0]['nu_p'][0]
    v = mcdc['materials'][0]['speed'][0]

    #Solve transport problem using QD given eddington tensors
    #build tridiagonal matrix for cell
    a = (Edd[:-1]/dx2/(1/v/dt+sigmaT))
    b = (Edd[1:]/dx2/(1/v/dt+sigmaT))
    f = np.zeros_like(x)
    f[1:-1] += dx*(1/v/dt+sigmaA-nu*sigmaF)
    d = np.zeros_like(x)
    d[1:-1] += dx*phi_in[1:-1]/v/dt

    #solve for scalar flux using thomas algorithm
    aa = -a
    bb = f
    bb[:-1] += a
    bb[1:] += b
    cc = -b
    dd = d
    #first row
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    #middle rows
    for i in range(Nx-1):
        cc[i+1]/= (bb[i+1]-aa[i]*cc[i])
        dd[i+1] = (dd[i+1]-aa[i]*dd[i])/(bb[i+1]-aa[i]*cc[i])
    #last row
    dd[Nx] = (dd[Nx]-aa[Nx-1]*dd[Nx-1])/(bb[Nx]-aa[Nx-1]*cc[Nx-1])

    #solve for phi
    phi = np.zeros_like(x)
    J = np.zeros_like(dx)
    #last row
    phi[Nx]=dd[Nx]
    #other rows
    for i in range(Nx):
        phi[Nx-1-i]=dd[Nx-1-i]-cc[Nx-1-i]*phi[Nx-i]

    #solve for current by multiplication
    for i in range(Nx):
        J[i] = (Edd[i]/dx[i]/(1/v/dt+sigmaT))*phi[i]-(Edd[i+1]/dx[i]/(1/v/dt+sigmaT))*phi[i+1]+J_in[i]

    return phi, J

#Step characteristics 1D transport
def step_char(mcdc, phi, J):
    
    t = mcdc['technique']['ww_mesh']['t']
    x = mcdc['technique']['ww_mesh']['x']
    dx = x[1:]-x[:-1]
    dt = t[mcdc['technique']['census_idx']+1]-t[mcdc['technique']['census_idx']]
    Nx = len(dx)
    sigmaT = mcdc['materials'][0]['total'][0]
    sigmaA = mcdc['materials'][0]['capture'][0]
    sigmaF = mcdc['materials'][0]['fission'][0]
    nu = mcdc['materials'][0]['nu'][0]
    v = mcdc['materials'][0]['speed'][0]
    

    return phi, J, Edd
