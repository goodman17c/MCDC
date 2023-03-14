import math

from mpi4py import MPI
from numba  import njit, objmode, literal_unroll

import matplotlib.pyplot as plt

import numpy as np


#Quasidiffusion
def QD1D(mcdc, phi_in, J_in, Edd_in):
    t = mcdc['technique']['ww_mesh']['t']
    x = mcdc['technique']['ww_mesh']['x']
    dx = x[1:]-x[:-1]
    Nx = len(dx)
    dx2 = np.zeros((Nx+1,))
    dx2[0] = dx[0]/2
    dx2[1:-1] = (dx[1:]+dx[:-1])/2
    dx2[-1] = dx[-1]/2
    dt = t[mcdc['technique']['census_idx']+1]-t[mcdc['technique']['census_idx']]
    sigmaT = mcdc['materials'][0]['total'][0]
    sigmaA = mcdc['materials'][0]['capture'][0]
    sigmaF = mcdc['materials'][0]['fission'][0]
    nu = mcdc['materials'][0]['nu_p'][0]
    v = mcdc['materials'][0]['speed'][0]

    Edd = np.zeros((Nx+2,))
    Edd[0] = Edd_in[0]
    Edd[1:-1] = Edd_in[:]
    Edd[-1] = Edd_in[-1]

    #Solve transport problem using QD given eddington tensors
    #build tridiagonal matrix for cell
    a = (Edd[:-1]/dx2/(1/v/dt+sigmaT))
    b = (Edd[1:]/dx2/(1/v/dt+sigmaT))
    f = np.zeros((Nx+2,))
    f[1:-1] += dx*(1/v/dt+sigmaA-nu*sigmaF)
    d = np.zeros((Nx+2,))
    d[1:-1] += dx*phi_in/v/dt

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
    for i in range(Nx):
        cc[i+1]/= (bb[i+1]-aa[i]*cc[i])
        dd[i+1] = (dd[i+1]-aa[i]*dd[i])/(bb[i+1]-aa[i]*cc[i])
    #last row
    dd[Nx+1] = (dd[Nx+1]-aa[Nx]*dd[Nx])/(bb[Nx+1]-aa[Nx]*cc[Nx])

    #solve for phi
    phi = np.zeros((Nx+2,))
    J = np.zeros((Nx+1,))

    #last row
    phi[Nx+1]=dd[Nx+1]
    #other rows
    for i in range(Nx+1):
        phi[Nx-i]=dd[Nx-i]-cc[Nx-i]*phi[Nx+1-i]

    #solve for current by multiplication
    for i in range(Nx+1):
        J[i] = (Edd[i]/dx2[i]/(1/v/dt+sigmaT))*phi[i]-(Edd[i+1]/dx2[i]/(1/v/dt+sigmaT))*phi[i+1]+J_in[i]

    
    res1 = np.zeros((Nx,))
    res2 = np.zeros((Nx+1,))
    #residuals      TODO need to add fision term
    for i in range(Nx):
        res1[i] = J[i+1]-J[i] + (sigmaA)*dx[i]*phi[i+1]
    for i in range(Nx+1):
        res2[i] = Edd[i+1]*phi[i+1]-Edd[i]*phi[i]+sigmaT*dx2[i]*J[i]
    print(np.amax(res1))
    print(np.argmax(res1))
    print(np.amax(res2))
    print(np.argmax(res2))


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
