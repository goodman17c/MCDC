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
    sigmaS = mcdc['materials'][0]['scatter'][0] + nu*sigmaF
    v = mcdc['materials'][0]['speed'][0]
    q=0

    dt2 = dt
    Edd2 = np.zeros((Nx+2,))
    Edd2[0] = Edd_in[0]
    Edd2[1:-1] = Edd_in[:]
    Edd2[-1] = Edd_in[-1]
    
    #0 BC
    phiLin=0
    phiRin=0
    JLin=0
    JRin=0
    CL=0
    CR=0


		#solve for phi
    phi = np.zeros((Nx,))
    phi_edge = np.zeros((Nx+1,))
    J = np.zeros((Nx+1,))
                
    #build tridiagonal matrix for cell
    a = (Edd2[:-1]/dx2/(1/v/dt2+sigmaT))
    b = (Edd2[1:]/dx2/(1/v/dt2+sigmaT))
    f = np.zeros((Nx+2,))
    f[0] = -CL
    f[1:-1] = dx*(sigmaT-sigmaS+1/v/dt2)
    f[-1] = CR
    d = np.zeros((Nx+2,))
    d[0] = JLin-CL*phiLin-J_in[0]/v/dt2/(sigmaT+1/v/dt2)
    d[1:-1] = dx*(q+phi_in[1:-1]/v/dt2)+(1/v/dt2)/(sigmaT+1/v/dt2)*(J_in[:-1]-J_in[1:])
    d[-1] = CR*phiRin-JRin+J[s,-1]/v/dt2/(sigmaT+1/v/dt2)
    #solve for scalar flux using thomas algorithm
    aa = np.copy(-a)
    bb = np.copy(f)
    bb[:-1] += a
    bb[1:] += b
    cc = np.copy(-b)
    dd = np.copy(d)
    #first row
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    #middle rows
    for i in range(Nx):
        cc[i+1] /= (bb[i+1]-aa[i]*cc[i])
        dd[i+1] = (dd[i+1]-aa[i]*dd[i])/(bb[i+1]-aa[i]*cc[i])
    #last row
    dd[Nx+1] = (dd[Nx+1]-aa[Nx]*dd[Nx])/(bb[Nx+1]-aa[Nx]*cc[Nx])

    #last row
    phi_edge[-1]=dd[Nx+1]
    phi[Nx-1]=dd[Nx]-cc[Nx]*phi_edge[-1]
    #other rows
    for i in range(Nx-1):
        phi[Nx-i-2]=dd[Nx-i-1]-cc[Nx-i-1]*phi[Nx-i-1]
    phi_edge[0]=dd[0]-cc[0]*phi[0]

    #solve for current by multiplication
    for i in range(Nx-1):
        J[i+1] = (Edd2[i+1]*phi[i]-Edd2[i+2]*phi[i+1]+J_in[i+1]*dx2[i+1]/v/dt2)/dx2[i+1]/(1/v/dt2+sigmaT)
    J[0] = (Edd2[0]*phi_edge[0]-Edd2[1]*phi[0]+J_in[0]*dx2[0]/v/dt2)/dx2[0]/(1/v/dt2+sigmaT)
    J[-1] = (Edd2[-2]*phi[-1]-Edd2[-1]*phi_edge[-1]+J_in[-1]*dx2[-1]/v/dt2)/dx2[-1]/(1/v/dt2+sigmaT)

    #residuals
    for i in range(Nx):
        res1[i] = J[i+1]-J[i] + (1/v/dt2+(sigmaT-sigmaS))*dx[i]*phi[i]-dx[i]/v/dt2*phi_in[i]-q*dx[i]
    
    res2[0] = Edd2[1]*phi[0]-Edd2[0]*phi_in[0]+sigmaT*dx2[0]*J[0]+dx2[0]/v/dt2*(J[0]-J_in[0])
    res2[-1] = Edd2[-1]*phi_in[-1]-Edd2[-2]*phi[-1]+sigmaT*dx2[-1]*J[-1]+dx2[-1]/v/dt2*(J[-1]-J_in[-1])
    for i in range(Nx-1):
        res2[i+1] = Edd2[i+2]*phi[i+1]-Edd2[i+1]*phi[i]+sigmaT*dx2[i+1]*J[i+1]+dx2[i+1]/v/dt2*(J[i+1]-J_in[i+1])

    return phi, J
