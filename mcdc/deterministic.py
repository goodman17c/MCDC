import numpy as np
from numba import njit


# Quasidiffusion
@njit
def QD1D(mcdc, dt, phi_in, J_in, Edd_in):
    t = mcdc["technique"]["ww_mesh"]["t"]
    x = mcdc["technique"]["ww_mesh"]["x"]
    x_mid = (x[1:] + x[:-1])/2
    dx = x[1:] - x[:-1]
    Nx = len(dx)
    dx2 = np.zeros((Nx + 1,))
    dx2[0] = dx[0] / 2
    dx2[1:-1] = (dx[1:] + dx[:-1]) / 2
    dx2[-1] = dx[-1] / 2
    sigmaT = mcdc["materials"][0]["total"][0]
    sigmaA = mcdc["materials"][0]["capture"][0]
    sigmaF = mcdc["materials"][0]["fission"][0]
    nu = mcdc["materials"][0]["nu_p"][0]
    sigmaS = mcdc["materials"][0]["scatter"][0] + nu * sigmaF
    v = mcdc["materials"][0]["speed"][0]
    q = 0

    dt2 = dt
    Edd2 = np.zeros((Nx + 2,))
    Edd2[0] = Edd_in[0]
    Edd2[1:-1] = Edd_in[:]
    Edd2[-1] = Edd_in[-1]

    # 0 BC
    phiLin = 0
    phiRin = 0
    JLin = 0
    JRin = 0
    CL = 0
    CR = 0

    # solve for phi
    phi = np.zeros((Nx,))
    phi_edge = np.zeros((Nx + 1,))
    J = np.zeros((Nx + 1,))

    res1 = np.zeros((Nx,))
    res2 = np.zeros((Nx + 1,))

    # build tridiagonal matrix for cell
    a = Edd2[:-1] / dx2 / (1 / v / dt2 + sigmaT)
    b = Edd2[1:] / dx2 / (1 / v / dt2 + sigmaT)
    f = np.zeros((Nx + 2,))
    f[0] = -CL
    f[1:-1] = dx * (sigmaT - sigmaS + 1 / v / dt2)
    f[-1] = CR
    d = np.zeros((Nx + 2,))
    d[0] = JLin - CL * phiLin - J_in[0] / v / dt2 / (sigmaT + 1 / v / dt2)
    d[1:-1] = dx * (q + phi_in[1:-1] / v / dt2) + (1 / v / dt2) / (
        sigmaT + 1 / v / dt2
    ) * (J_in[:-1] - J_in[1:])
    d[-1] = CR * phiRin - JRin + J_in[-1] / v / dt2 / (sigmaT + 1 / v / dt2)
    # solve for scalar flux using thomas algorithm
    aa = np.copy(-a)
    bb = np.copy(f)
    bb[:-1] += a
    bb[1:] += b
    cc = np.copy(-b)
    dd = np.copy(d)
    # first row
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    # middle rows
    for i in range(Nx):
        cc[i + 1] /= bb[i + 1] - aa[i] * cc[i]
        dd[i + 1] = (dd[i + 1] - aa[i] * dd[i]) / (bb[i + 1] - aa[i] * cc[i])
    # last row
    dd[Nx + 1] = (dd[Nx + 1] - aa[Nx] * dd[Nx]) / (bb[Nx + 1] - aa[Nx] * cc[Nx])

    # last row
    phi_edge[-1] = dd[Nx + 1]
    phi[Nx - 1] = dd[Nx] - cc[Nx] * phi_edge[-1]
    # other rows
    for i in range(Nx - 1):
        phi[Nx - i - 2] = dd[Nx - i - 1] - cc[Nx - i - 1] * phi[Nx - i - 1]
    phi_edge[0] = dd[0] - cc[0] * phi[0]

    # solve for current by multiplication
    for i in range(Nx - 1):
        J[i + 1] = (
            (
                Edd2[i + 1] * phi[i]
                - Edd2[i + 2] * phi[i + 1]
                + J_in[i + 1] * dx2[i + 1] / v / dt2
            )
            / dx2[i + 1]
            / (1 / v / dt2 + sigmaT)
        )
    J[0] = (
        (Edd2[0] * phi_edge[0] - Edd2[1] * phi[0] + J_in[0] * dx2[0] / v / dt2)
        / dx2[0]
        / (1 / v / dt2 + sigmaT)
    )
    J[-1] = (
        (Edd2[-2] * phi[-1] - Edd2[-1] * phi_edge[-1] + J_in[-1] * dx2[-1] / v / dt2)
        / dx2[-1]
        / (1 / v / dt2 + sigmaT)
    )

    # residuals
    for i in range(Nx):
        res1[i] = (
            J[i + 1]
            - J[i]
            + (1 / v / dt2 + (sigmaT - sigmaS)) * dx[i] * phi[i]
            - dx[i] / v / dt2 * phi_in[i + 1]
            - q * dx[i]
        )

    res2[0] = (
        Edd2[1] * phi[0]
        - Edd2[0] * phi_in[0]
        + sigmaT * dx2[0] * J[0]
        + dx2[0] / v / dt2 * (J[0] - J_in[0])
    )
    res2[-1] = (
        Edd2[-1] * phi_in[-1]
        - Edd2[-2] * phi[-1]
        + sigmaT * dx2[-1] * J[-1]
        + dx2[-1] / v / dt2 * (J[-1] - J_in[-1])
    )
    for i in range(Nx - 1):
        res2[i + 1] = (
            Edd2[i + 2] * phi[i + 1]
            - Edd2[i + 1] * phi[i]
            + sigmaT * dx2[i + 1] * J[i + 1]
            + dx2[i + 1] / v / dt2 * (J[i + 1] - J_in[i + 1])
        )

    # Remove file if beginning simulation
    if (mcdc["technique"]["census_idx"] == 0 and mcdc["i_cycle"] == 0):
        with open(mcdc["setting"]["output"] + "_det.txt", "w") as f:
            f.write("")

    # Write an output file result from each iteration
    with open(mcdc["setting"]["output"] + "_det.txt", "a") as f:
        # Write time step and iteration
        f.write("Time Step " + str(mcdc["technique"]["census_idx"] + 1 + int((mcdc["i_cycle"]+1)/(mcdc["technique"]["updates"]+1)) ))
        f.write(" Update "+ str((mcdc["i_cycle"] + 1)%(mcdc["technique"]["updates"]+1)) + "\n")
        f.write("Max balance equation residual: %16.5e\n"%np.max(res1))
        f.write("Max 1st moment equation residual: %16.5e\n"%np.max(res2))

        # Write the inputs and outputs
        f.write("       x_mid")
        f.write("   Scalar flux 0")
        f.write("     Scalar flux")
        f.write("       Eddington")
        f.write("           x")
        f.write("       Current 0")
        f.write("         Current")
        f.write("\n")
        for i in range(Nx):
            f.write("%12.3f%16.5e%16.5e%16.5e%12.3f%16.5e%16.5e\n"%(x_mid[i], phi_in[i+1], phi[i], Edd_in[i], x[i], J_in[i], J[i]))
        f.write("%72.3f%16.5e%16.5e\n"%(x[Nx], J_in[Nx], J[Nx]))
        f.write("\n")

    return phi, J

# Quasidiffusion Crank-Nicholson
@njit
def QD1DCN(mcdc, phi_in, J_in, Edd_in, Edd_in0):
    t = mcdc["technique"]["ww_mesh"]["t"]
    x = mcdc["technique"]["ww_mesh"]["x"]
    x_mid = (x[1:] + x[:-1])/2
    dx = x[1:] - x[:-1]
    Nx = len(dx)
    dx2 = np.zeros((Nx + 1,))
    dx2[0] = dx[0] / 2
    dx2[1:-1] = (dx[1:] + dx[:-1]) / 2
    dx2[-1] = dx[-1] / 2
    sigmaT = mcdc["materials"][0]["total"][0]
    sigmaA = mcdc["materials"][0]["capture"][0]
    sigmaF = mcdc["materials"][0]["fission"][0]
    nu = mcdc["materials"][0]["nu_p"][0]
    sigmaS = mcdc["materials"][0]["scatter"][0] + nu * sigmaF
    v = mcdc["materials"][0]["speed"][0]
    q = 0

    dt2 = dt
    Edd2 = np.zeros((Nx + 2,))
    Edd2[0] = Edd_in[0]
    Edd2[1:-1] = Edd_in[:]
    Edd2[-1] = Edd_in[-1]
    
    Edd2old = np.zeros((Nx + 2,))
    Edd2old[0] = Edd_in0[0]
    Edd2old[1:-1] = Edd_in0[:]
    Edd2old[-1] = Edd_in0[-1]

    # 0 BC
    phiLin = 0
    phiRin = 0
    JLin = 0
    JRin = 0
    CL = 0
    CR = 0

    # solve for phi
    phi = np.zeros((Nx,))
    phi_edge = np.zeros((Nx + 1,))
    J = np.zeros((Nx + 1,))

    res1 = np.zeros((Nx,))
    res2 = np.zeros((Nx + 1,))

    # build tridiagonal matrix for cell
    a = Edd2[:-1] / dx2 / (1 / v / dt2 + sigmaT)
    b = Edd2[1:] / dx2 / (1 / v / dt2 + sigmaT)
    f = np.zeros((Nx + 2,))
    f[0] = -CL
    f[1:-1] = dx * (sigmaT - sigmaS + 1 / v / dt2)
    f[-1] = CR
    d = np.zeros((Nx + 2,))
    d[0] = JLin - CL * phiLin - J_in[0] / v / dt2 / (sigmaT + 1 / v / dt2)
    d[1:-1] = dx * (q + phi_in[1:-1] / v / dt2) + (1 / v / dt2) / (
        sigmaT + 1 / v / dt2
    ) * (J_in[:-1] - J_in[1:])
    d[-1] = CR * phiRin - JRin + J_in[-1] / v / dt2 / (sigmaT + 1 / v / dt2)
    # solve for scalar flux using thomas algorithm
    aa = np.copy(-a)
    bb = np.copy(f)
    bb[:-1] += a
    bb[1:] += b
    cc = np.copy(-b)
    dd = np.copy(d)
    # first row
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    # middle rows
    for i in range(Nx):
        cc[i + 1] /= bb[i + 1] - aa[i] * cc[i]
        dd[i + 1] = (dd[i + 1] - aa[i] * dd[i]) / (bb[i + 1] - aa[i] * cc[i])
    # last row
    dd[Nx + 1] = (dd[Nx + 1] - aa[Nx] * dd[Nx]) / (bb[Nx + 1] - aa[Nx] * cc[Nx])

    # last row
    phi_edge[-1] = dd[Nx + 1]
    phi[Nx - 1] = dd[Nx] - cc[Nx] * phi_edge[-1]
    # other rows
    for i in range(Nx - 1):
        phi[Nx - i - 2] = dd[Nx - i - 1] - cc[Nx - i - 1] * phi[Nx - i - 1]
    phi_edge[0] = dd[0] - cc[0] * phi[0]

    # solve for current by multiplication
    for i in range(Nx - 1):
        J[i + 1] = (
            (
                Edd2[i + 1] * phi[i]
                - Edd2[i + 2] * phi[i + 1]
                + J_in[i + 1] * dx2[i + 1] / v / dt2
            )
            / dx2[i + 1]
            / (1 / v / dt2 + sigmaT)
        )
    J[0] = (
        (Edd2[0] * phi_edge[0] - Edd2[1] * phi[0] + J_in[0] * dx2[0] / v / dt2)
        / dx2[0]
        / (1 / v / dt2 + sigmaT)
    )
    J[-1] = (
        (Edd2[-2] * phi[-1] - Edd2[-1] * phi_edge[-1] + J_in[-1] * dx2[-1] / v / dt2)
        / dx2[-1]
        / (1 / v / dt2 + sigmaT)
    )

    # residuals
    for i in range(Nx):
        res1[i] = (
            J[i + 1]
            - J[i]
            + (1 / v / dt2 + (sigmaT - sigmaS)) * dx[i] * phi[i]
            - dx[i] / v / dt2 * phi_in[i + 1]
            - q * dx[i]
        )

    res2[0] = (
        Edd2[1] * phi[0]
        - Edd2[0] * phi_in[0]
        + sigmaT * dx2[0] * J[0]
        + dx2[0] / v / dt2 * (J[0] - J_in[0])
    )
    res2[-1] = (
        Edd2[-1] * phi_in[-1]
        - Edd2[-2] * phi[-1]
        + sigmaT * dx2[-1] * J[-1]
        + dx2[-1] / v / dt2 * (J[-1] - J_in[-1])
    )
    for i in range(Nx - 1):
        res2[i + 1] = (
            Edd2[i + 2] * phi[i + 1]
            - Edd2[i + 1] * phi[i]
            + sigmaT * dx2[i + 1] * J[i + 1]
            + dx2[i + 1] / v / dt2 * (J[i + 1] - J_in[i + 1])
        )

    # Remove file if beginning simulation
    if (mcdc["technique"]["census_idx"] == 0 and mcdc["i_cycle"] == 0):
        with open(mcdc["setting"]["output"] + "_det.txt", "w") as f:
            f.write("")

    # Write an output file result from each iteration
    with open(mcdc["setting"]["output"] + "_det.txt", "a") as f:
        # Write time step and iteration
        f.write("Time Step " + str(mcdc["technique"]["census_idx"] + 1 + int((mcdc["i_cycle"]+1)/(mcdc["technique"]["updates"]+1)) ))
        f.write(" Update "+ str((mcdc["i_cycle"] + 1)%(mcdc["technique"]["updates"]+1)) + "\n")
        f.write("Max balance equation residual: %16.5e\n"%np.max(res1))
        f.write("Max 1st moment equation residual: %16.5e\n"%np.max(res2))

        # Write the inputs and outputs
        f.write("       x_mid")
        f.write("   Scalar flux 0")
        f.write("     Scalar flux")
        f.write("       Eddington")
        f.write("           x")
        f.write("       Current 0")
        f.write("         Current")
        f.write("\n")
        for i in range(Nx):
            f.write("%12.3f%16.5e%16.5e%16.5e%12.3f%16.5e%16.5e\n"%(x_mid[i], phi_in[i+1], phi[i], Edd_in[i], x[i], J_in[i], J[i]))
        f.write("%72.3f%16.5e%16.5e\n"%(x[Nx], J_in[Nx], J[Nx]))
        f.write("\n")

    return phi, J
