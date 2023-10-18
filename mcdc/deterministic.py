import numpy as np
from numba import njit

# Naive Eddington Fixup
@njit
def EddFix(Edd_in):
    Edd_out = np.copy(Edd_in)
    Edd_out[np.isnan(Edd_in)] = 0.33
    Edd_out[Edd_in<1E-2] = 1E-2
    return Edd_out

# Linear Eddington Interpolator
@njit
def LinearEdd(Edd_in):
    Edd_in[Edd_in<1E-2] = 1E-2
    done = ~np.isnan(Edd_in)
    if np.count_nonzero(~done) == 0:
        return Edd_in
    Edd_out = np.full_like(Edd_in, 1 / 3)
    Edd_out[~np.isnan(Edd_in)] = Edd_in[~np.isnan(Edd_in)]
    xleft = np.nonzero(done)[0][0]
    xright = np.nonzero(done)[0][-1]
    #Edd_out[:xleft] = Edd_out[xleft]
    #Edd_out[xright:] = Edd_out[xright]
    done[:xleft] = True
    done[xright:] = True
    for i in range(xleft, xright):
        if not done[i]:
            xright2 = np.nonzero(done[i : xright + 1])[0][0]
            Edd_out[i] = (
                Edd_out[i - 1] * xright2 + Edd_in[i + xright2]
            ) / (1 + xright2)
            done[i] = True
    return Edd_out

# Linear SF Interpolator
@njit
def LinearSF(phi_in):
    Nx = len(phi_in)
    if np.count_nonzero(phi_in==0) == 0:
        return phi_in
    if np.count_nonzero(phi_in) == 0:
        return phi_in
    phi_out = phi_in
    xleft = np.nonzero(phi_in)[0][0]
    xright = np.nonzero(phi_in)[0][-1]
    for i in range(xleft, xright):
        if phi_in[i+1] == 0:
            xright2 = np.nonzero(phi_in[i + 1 : xright + 1])[0][0]
            phi_out[i+1] = (
                phi_out[i] * xright2 + phi_in[i + 1 +  xright2]
            ) / (1 + xright2)
    return phi_out

# Current Reconstruction - cell balance
@njit
def JCellBal(phi, phi_old, sigmaA, vdt, dx):
    print(phi)
    print(phi_old)
    Nx = len(dx)
    dJ = -(sigmaA + 1/vdt)*dx*phi + (phi_old/vdt)*dx
    J = np.zeros((Nx+1,))
    J[1:] = np.cumsum(dJ)
    return J

# Eddington Interpolator
@njit
def EddInterp(Edd_in, n_in=None, d=0, l=0, p=0):
    if n_in is None:
        n_in = np.ones_like(Edd_in)
    n = n_in + l
    Edd_in[np.isnan(Edd_in)] = 0
    Edd = Edd_in * n_in + l / 3
    for j in range(d):
        n[: -1 - j] += n_in[j + 1 :] * (j + 2) ** (-p)
        n[j + 1 :] += n_in[: -1 - j] * (j + 2) ** (-p)
        Edd[: -1 - j] += n_in[j + 1 :] * Edd_in[j + 1 :] * (j + 2) ** (-p)
        Edd[j + 1 :] += n_in[: -1 - j] * Edd_in[: -1 - j] * (j + 2) ** (-p)
    Edd /= n
    return Edd

# Naive Flux Fixup
@njit
def SFFix(phi_in):
    #tent blur
    #d = 3
    phi_out = np.copy(phi_in)
    phi_out *= 6/16
    phi_out[:-1] += phi_in[1:]*3/16
    phi_out[1:] += phi_in[:-1]*3/16
    phi_out[:-2] += phi_in[2:]/16
    phi_out[2:] += phi_in[:-2]/16

    return phi_out

# Naive Current Fixup
@njit
def JFix(phi_in, J_in, dxvdt):
    Nx = len(dxvdt)
    mid = int(Nx/2)
    J_out = np.copy(J_in)
    tmp = dxvdt[mid]*phi_in[mid]
    for i in range(mid):
        if J_out[mid+i] > tmp:
            J_out[mid+i] = tmp
        tmp = J_out[mid+i]+dxvdt[mid+i]*phi_in[mid+i]/2
    tmp = -dxvdt[mid]*phi_in[mid]
    for i in range(mid):
        if J_out[mid-1-i] < tmp:
            J_out[mid-1-i] = tmp
        tmp = J_out[mid-1-i]-dxvdt[mid-1-i]*phi_in[mid-1-i]/2
    print(np.max(np.abs(J_out-J_in))/np.max(np.abs(J_in)))
    return J_out

# Quasidiffusion
@njit
def QD1D(mcdc, dt, phi_in, J_in, Edd_raw):
    x = mcdc["technique"]["ww_mesh"]["x"]
    x_mid = (x[1:] + x[:-1]) / 2
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

    Edd_in = LinearEdd(Edd_raw)
    phi_in = LinearSF(phi_in)
    J_in = LinearSF(J_in)
    #Edd_in = EddFix(Edd_raw)

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
    a = Edd2[:-1] / dx2 / (1 / v / dt + sigmaT)
    b = Edd2[1:] / dx2 / (1 / v / dt + sigmaT)
    f = np.zeros((Nx + 2,))
    f[0] = -CL
    f[1:-1] = dx * (sigmaT - sigmaS + 1 / v / dt)
    f[-1] = CR
    d = np.zeros((Nx + 2,))
    d[0] = JLin - CL * phiLin - J_in[0] / v / dt / (sigmaT + 1 / v / dt)
    d[1:-1] = dx * (q + phi_in[1:-1] / v / dt) + (1 / v / dt) / (
        sigmaT + 1 / v / dt
    ) * (J_in[:-1] - J_in[1:])
    d[-1] = CR * phiRin - JRin + J_in[-1] / v / dt / (sigmaT + 1 / v / dt)
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
                + J_in[i + 1] * dx2[i + 1] / v / dt
            )
            / dx2[i + 1]
            / (1 / v / dt + sigmaT)
        )
    J[0] = (
        (Edd2[0] * phi_edge[0] - Edd2[1] * phi[0] + J_in[0] * dx2[0] / v / dt)
        / dx2[0]
        / (1 / v / dt + sigmaT)
    )
    J[-1] = (
        (Edd2[-2] * phi[-1] - Edd2[-1] * phi_edge[-1] + J_in[-1] * dx2[-1] / v / dt)
        / dx2[-1]
        / (1 / v / dt + sigmaT)
    )

    # residuals
    for i in range(Nx):
        res1[i] = (
            J[i + 1]
            - J[i]
            + (1 / v / dt + (sigmaT - sigmaS)) * dx[i] * phi[i]
            - dx[i] / v / dt * phi_in[i + 1]
            - q * dx[i]
        )

    res2[0] = (
        Edd2[1] * phi[0]
        - Edd2[0] * phi_in[0]
        + sigmaT * dx2[0] * J[0]
        + dx2[0] / v / dt * (J[0] - J_in[0])
    )
    res2[-1] = (
        Edd2[-1] * phi_in[-1]
        - Edd2[-2] * phi[-1]
        + sigmaT * dx2[-1] * J[-1]
        + dx2[-1] / v / dt * (J[-1] - J_in[-1])
    )
    for i in range(Nx - 1):
        res2[i + 1] = (
            Edd2[i + 2] * phi[i + 1]
            - Edd2[i + 1] * phi[i]
            + sigmaT * dx2[i + 1] * J[i + 1]
            + dx2[i + 1] / v / dt * (J[i + 1] - J_in[i + 1])
        )

    # Remove file if beginning simulation
    if mcdc["technique"]["census_idx"] == 0 and mcdc["i_cycle"] == 0:
        with open(mcdc["setting"]["output"] + "_det.txt", "w") as f:
            f.write("")

    # Write an output file result from each iteration
    with open(mcdc["setting"]["output"] + "_det.txt", "a") as f:
        # Write time step and iteration
        f.write(
            "Time Step "
            + str(
                mcdc["technique"]["census_idx"]
                + 1
                + int((mcdc["i_cycle"] + 1) / (mcdc["technique"]["updates"] + 1))
            )
        )
        f.write(
            " Update "
            + str((mcdc["i_cycle"] + 1) % (mcdc["technique"]["updates"] + 1))
            + "\n"
        )
        f.write("Max balance equation residual: %16.5e\n" % np.max(res1))
        f.write("Max 1st moment equation residual: %16.5e\n" % np.max(res2))

        # Write the inputs and outputs
        f.write("       x_mid")
        f.write("   Scalar flux 0")
        f.write("     Scalar flux")
        f.write("       Eddington")
        f.write("      Interp VEF")
        f.write("           x")
        f.write("       Current 0")
        f.write("         Current")
        f.write("\n")
        for i in range(Nx):
            f.write(
                "%12.3f%16.5e%16.5e%16.5e%16.5e%12.3f%16.5e%16.5e\n"
                % (
                    x_mid[i],
                    phi_in[i + 1],
                    phi[i],
                    Edd_raw[i],
                    Edd_in[i],
                    x[i],
                    J_in[i],
                    J[i],
                )
            )
        f.write("%72.3f%16.5e%16.5e\n" % (x[Nx], J_in[Nx], J[Nx]))
        f.write("\n")

    return phi, J


# Quasidiffusion Crank-Nicholson
@njit
def QD1DCN(mcdc, phi_in, J_in, Edd_raw, Edd_raw0):
    t = mcdc["technique"]["census_idx"]
    t2 = mcdc["technique"]["ww_mesh"]["t"]
    x = mcdc["technique"]["ww_mesh"]["x"]
    x_mid = (x[1:] + x[:-1]) / 2

    dx = x[1:] - x[:-1]
    dt = t2[t + 1] - t2[t]
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

    #Edd_in = EddFix(Edd_raw)
    #Edd_in0 = EddFix(Edd_raw0)
    Edd_in = LinearEdd(Edd_raw)
    Edd_in0 = LinearEdd(Edd_raw0)
    phi_in = LinearSF(phi_in)
    J_in = LinearSF(J_in)

    Edd2 = np.zeros((Nx + 2,))
    Edd2[0] = Edd_in[0]
    Edd2[1:-1] = Edd_in[:]
    Edd2[-1] = Edd_in[-1]

    Edd2old = np.zeros((Nx + 2,))
    Edd2old[0] = Edd_in0[0]
    Edd2old[1:-1] = Edd_in0[:]
    Edd2old[-1] = Edd_in0[-1]

    vdt = v * dt
    sigmaTt = sigmaT + 2 / vdt

    # 0 BC
    phiLin = 0
    phiRin = 0
    JLin = 0
    JRin = 0
    CL = 0.5
    CR = 0.5

    # solve for phi
    phi = np.zeros((Nx,))
    phi_edge = np.zeros((Nx + 1,))
    J = np.zeros((Nx + 1,))
    sol = np.zeros((2 * Nx + 3,))

    res1 = np.zeros((Nx,))
    res2 = np.zeros((Nx + 1,))

    # build tridiagonal matrix for cell
    aa = np.zeros((2 * Nx + 2,))
    bb = np.zeros((2 * Nx + 3,))
    cc = np.zeros((2 * Nx + 2,))
    dd = np.zeros((2 * Nx + 3,))
    # 1st moment equation
    aa[0:-1:2] = -Edd2[:-1]
    cc[1::2] = Edd2[1:]
    bb[1:-1:2] = sigmaTt * dx2
    dd[1:-1:2] = (
        Edd2old[:-1] * phi_in[:-1]
        - Edd2old[1:] * phi_in[1:]
        - dx2 * (sigmaT - 2 / vdt) * J_in
    )
    # Balance equation
    aa[1:-2:2] = -1
    cc[2:-1:2] = 1
    bb[2:-2:2] = (sigmaTt - sigmaS) * dx
    dd[2:-2:2] = 2 * dx * q - (sigmaT - sigmaS - 2 / vdt) * dx * phi_in[1:-1] - J_in[1:] + J_in[:-1]
    # Boundary Conditions
    bb[0] = CL
    cc[0] = 1
    dd[0] = CL*phiLin+JLin
    bb[-1] = CR
    aa[-1] = -1
    dd[-1] = CR*phiRin-JRin


    # first row
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    # middle rows
    for i in range(2 * Nx + 1):
        cc[i + 1] /= bb[i + 1] - aa[i] * cc[i]
        dd[i + 1] = (dd[i + 1] - aa[i] * dd[i]) / (bb[i + 1] - aa[i] * cc[i])
    # last row
    dd[2 * Nx + 2] = (dd[2 * Nx + 2] - aa[2 * Nx + 1] * dd[2 * Nx + 1]) / (
        bb[2 * Nx + 2] - aa[2 * Nx + 1] * cc[2 * Nx + 1]
    )

    # last row
    sol[2 * Nx + 2] = dd[2 * Nx + 2]
    # other rows
    for i in range(2 * Nx + 2):
        sol[2 * Nx + 1 - i] = (
            dd[2 * Nx + 1 - i] - cc[2 * Nx + 1 - i] * sol[2 * Nx + 2 - i]
        )

    J = sol[1:-1:2]
    phi = sol[2:-2:2]
    phi_edge[-1] = sol[-1]
    phi_edge[0] = sol[0]

    # residuals
    for i in range(Nx):
        res1[i] = (
            (J[i + 1] + J_in[i + 1] - J[i] - J_in[i]) / 2
            + (sigmaT - sigmaS) * dx[i] * (phi[i]+phi_in[i + 1]) / 2
            - dx[i] / vdt * (phi[i] - phi_in[i + 1])
            - q * dx[i]
        )

    res2[0] = (
        (Edd2[1] * phi[0] + Edd2old[1] * phi_in[1]
        - Edd2[0] * phi_edge[0] - Edd2old[0] * phi_in[0])/2
        + sigmaT * dx2[0] * (J[0] + J_in[0]) / 2
        + dx2[0] / vdt * (J[0] - J_in[0])
    )
    res2[-1] = (
        (Edd2old[-1] * phi_in[-1] + Edd2[-1] * phi_edge[-1]
        - Edd2[-2] * phi[-1] - Edd2old[-2] * phi_in[-2])/2
        + sigmaT * dx2[-1] * (J[-1] + J_in[-1]) / 2
        + dx2[-1] / vdt * (J[-1] - J_in[-1])
    )
    for i in range(Nx - 1):
        res2[i + 1] = (
            (Edd2[i + 2] * phi[i + 1] + Edd2old [i + 2] * phi_in[i + 2]
            - Edd2[i + 1] * phi[i] - Edd2old[i + 1] * phi_in[i + 1]) / 2
            + sigmaT * dx2[i + 1] * (J[i + 1] + J_in[i+1]) / 2
            + dx2[i + 1] / vdt * (J[i + 1] - J_in[i + 1])
        )

    # Remove file if beginning simulation
    if mcdc["technique"]["census_idx"] == 0 and mcdc["i_cycle"] == 0:
        with open(mcdc["setting"]["output"] + "_det.txt", "w") as f:
            f.write("")

    # Write an output file result from each iteration
    with open(mcdc["setting"]["output"] + "_det.txt", "a") as f:
        # Write time step and iteration
        f.write(
            "Time Step "
            + str(
                mcdc["technique"]["census_idx"]
                + 1
                + int((mcdc["i_cycle"] + 1) / (mcdc["technique"]["updates"] + 1))
            )
        )
        f.write(
            " Update "
            + str((mcdc["i_cycle"] + 1) % (mcdc["technique"]["updates"] + 1))
            + "\n"
        )
        f.write("Max balance equation residual: %16.5e\n" % np.max(res1))
        f.write("Max 1st moment equation residual: %16.5e\n" % np.max(res2))

        # Write the inputs and outputs
        f.write("       x_mid")
        f.write("   Scalar flux 0")
        f.write("     Scalar flux")
        f.write("       Eddington")
        f.write("      Interp VEF")
        f.write("           x")
        f.write("       Current 0")
        f.write("         Current")
        f.write("\n")
        for i in range(Nx):
            f.write(
                "%12.3f%16.5e%16.5e%16.5e%16.5e%12.3f%16.5e%16.5e\n"
                % (
                    x_mid[i],
                    phi_in[i + 1],
                    phi[i],
                    Edd_raw[i],
                    Edd_in[i],
                    x[i],
                    J_in[i],
                    J[i],
                )
            )
        f.write("%72.3f%16.5e%16.5e\n" % (x[Nx], J_in[Nx], J[Nx]))
        f.write("\n")

    return phi, J

# Forward Euler
@njit
def QD1DFE(mcdc, dt, phi_in, J_in, Edd_raw, phi_old):
    x = mcdc["technique"]["ww_mesh"]["x"]
    x_mid = (x[1:] + x[:-1]) / 2

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

    Edd_in = LinearEdd(Edd_raw)
    Edd2 = np.zeros((Nx + 2,))
    Edd2[0] = Edd_in[0]
    Edd2[1:-1] = Edd_in[:]
    Edd2[-1] = Edd_in[-1]

    vdt = v * dt

    dxvdt = dx/vdt

    #phi_in = SFFix(phi_in)
    #J_tmp = J_in
    phi_in = LinearSF(phi_in)
    phi_old = LinearSF(phi_old)
    J_in = LinearSF(J_in)
    #J_in = JCellBal(phi_in[1:-1], phi_old, sigmaT-sigmaS-nu*sigmaF, v*dt, dx)

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

    phi = phi_in[1:-1] - vdt * ((J_in[1:]-J_in[:-1])/dx + (sigmaT-sigmaS)*phi_in[1:-1] - q)
    J = J_in - vdt*((Edd2[1:]*phi_in[1:] - Edd2[:-1]*phi_in[:-1])/dx2 + sigmaT*J_in)

    # residuals
    for i in range(Nx):
        res1[i] = (
            (J_in[i + 1] - J_in[i])
            + (sigmaT - sigmaS) * dx[i] * phi_in[i + 1]
            + dx[i] / vdt * (phi[i] - phi_in[i + 1])
            - q * dx[i]
        )

    res2[0] = (
        Edd2[1] * phi_in[1] - Edd2[0] * phi_in[0]
        + sigmaT * dx2[0] * J_in[0]
        + dx2[0] / vdt * (J[0] - J_in[0])
    )
    res2[-1] = (
        Edd2[-1] * phi_in[-1] - Edd2[-2] * phi_in[-2]
        + sigmaT * dx2[-1] * J_in[-1]
        + dx2[-1] / vdt * (J[-1] - J_in[-1])
    )
    for i in range(Nx - 1):
        res2[i + 1] = (
            Edd2[i + 2] * phi_in[i + 2] - Edd2[i + 1] * phi_in[i+1]
            + sigmaT * dx2[i + 1] * J_in[i+1]
            + dx2[i + 1] / vdt * (J[i + 1] - J_in[i + 1])
        )

    # Remove file if beginning simulation
    if mcdc["technique"]["census_idx"] == 0 and mcdc["i_cycle"] == 0:
        with open(mcdc["setting"]["output"] + "_det.txt", "w") as f:
            f.write("")

    # Write an output file result from each iteration
    with open(mcdc["setting"]["output"] + "_det.txt", "a") as f:
        # Write time step and iteration
        f.write(
            "Time Step "
            + str(
                mcdc["technique"]["census_idx"]
                + 1
                + int((mcdc["i_cycle"] + 1) / (mcdc["technique"]["updates"] + 1))
            )
        )
        f.write(
            " Update "
            + str((mcdc["i_cycle"] + 1) % (mcdc["technique"]["updates"] + 1))
            + "\n"
        )
        f.write("Max balance equation residual: %16.5e\n" % np.max(res1))
        f.write("Max 1st moment equation residual: %16.5e\n" % np.max(res2))

        # Write the inputs and outputs
        f.write("       x_mid")
        f.write("   Scalar flux 0")
        f.write("     Scalar flux")
        f.write("       Eddington")
        f.write("      Interp VEF")
        f.write("           x")
        f.write("       Current 0")
        f.write("         Current")
        f.write("\n")
        for i in range(Nx):
            f.write(
                "%12.3f%16.5e%16.5e%16.5e%16.5e%12.3f%16.5e%16.5e\n"
                % (
                    x_mid[i],
                    phi_in[i + 1],
                    phi[i],
                    Edd_raw[i],
                    Edd_in[i],
                    x[i],
                    J_in[i],
                    J[i],
                )
            )
        f.write("%72.3f%16.5e%16.5e\n" % (x[Nx], J_in[Nx], J[Nx]))
        f.write("\n")

    return phi, J

# Runga-Kutta Explicit Improved Euler Heun's Method
@njit
def QD1DHeun(mcdc, phi_in, J_in, Edd_raw, Edd_raw0, phi_old):
    t = mcdc["technique"]["census_idx"]
    t2 = mcdc["technique"]["ww_mesh"]["t"]
    x = mcdc["technique"]["ww_mesh"]["x"]
    x_mid = (x[1:] + x[:-1]) / 2

    dx = x[1:] - x[:-1]
    dt = t2[t + 1] - t2[t]
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

    Edd_in = EddFix(Edd_raw)
    Edd_in0 = EddFix(Edd_raw0)
    Edd2 = np.zeros((Nx + 2,))
    Edd2[0] = Edd_in[0]
    Edd2[1:-1] = Edd_in[:]
    Edd2[-1] = Edd_in[-1]
    
    Edd2old = np.zeros((Nx + 2,))
    Edd2old[0] = Edd_in0[0]
    Edd2old[1:-1] = Edd_in0[:]
    Edd2old[-1] = Edd_in0[-1]

    vdt = v * dt
    sigmaTt = sigmaT + 2 / vdt
    
    J_in = JCellBal(phi_old, phi_in[1:-1], sigmaT-sigmaS-nu*sigmaF, v*dt, dx)

    # 0 BC
    phiLin = 0
    phiRin = 0
    JLin = 0
    JRin = 0
    CL = -0.5
    CR = 0.5
    
    # solve for phi
    phi2 = np.zeros((Nx+2,))
    J = np.zeros((Nx + 1,))

    res1 = np.zeros((Nx,))
    res2 = np.zeros((Nx + 1,))

    phi2[1:-1] = phi_in[1:-1] - vdt/2 * ((J_in[1:]-J_in[:-1])/dx + (sigmaT-sigmaS)*phi_in[1:-1] - q)
    J2 = J_in - vdt/2*((Edd2old[1:]*phi_in[1:] - Edd2old[:-1]*phi_in[:-1])/dx2 + sigmaT*J_in)
    phi2[0] = (J2[0]-JLin)/CL + phiLin
    phi2[-1] = (J2[-1]-JRin)/CR + phiRin
    
    phi = phi_in[1:-1] - vdt/2 * ((J_in[1:]-J_in[:-1])/dx + (sigmaT-sigmaS)*phi_in[1:-1] - q +
    (J2[1:]-J2[:-1])/dx + (sigmaT-sigmaS)*phi2[1:-1] - q)
    J = J_in - vdt/2*((Edd2old[1:]*phi_in[1:] - Edd2old[:-1]*phi_in[:-1])/dx2 + sigmaT*J_in + (Edd2[1:]*phi2[1:] - Edd2[:-1]*phi2[:-1])/dx2 + sigmaT*J2)

    # residuals
    for i in range(Nx):
        res1[i] = (
            (J_in[i + 1] - J_in[i] + J2[i+1] - J2[i])/2
            + (sigmaT - sigmaS) * dx[i] * (phi_in[i + 1] + phi2[i+1])/2
            + dx[i] / vdt * (phi[i] - phi_in[i + 1])
            - q * dx[i]
        )

    res2[0] = (
        (Edd2old[1] * phi_in[1] - Edd2old[0] * phi_in[0]
        + Edd2[1] * phi2[1] - Edd2[0] * phi2[0])/2
        + sigmaT * dx2[0] * (J2[0] + J_in[0])/2
        + dx2[0] / vdt * (J[0] - J_in[0])
    )
    res2[-1] = (
        (Edd2old[-1] * phi_in[-1] - Edd2old[-2] * phi_in[-2]
        + Edd2[-1] * phi2[-1] - Edd2[-2] * phi2[-2])/2
        + sigmaT * dx2[-1] * (J2[-1] + J_in[-1])/2
        + dx2[-1] / vdt * (J[-1] - J_in[-1])
    )
    for i in range(Nx - 1):
        res2[i + 1] = (
            (Edd2old[i + 2] * phi_in[i + 2] - Edd2old[i + 1] * phi_in[i+1]
            + Edd2[i + 2] * phi2[i + 2] - Edd2[i + 1] * phi2[i+1])/2
            + sigmaT * dx2[i + 1] * (J2[i+1] + J_in[i+1])/2
            + dx2[i + 1] / vdt * (J[i + 1] - J_in[i + 1])
        )

    # Remove file if beginning simulation
    if mcdc["technique"]["census_idx"] == 0 and mcdc["i_cycle"] == 0:
        with open(mcdc["setting"]["output"] + "_det.txt", "w") as f:
            f.write("")

    # Write an output file result from each iteration
    with open(mcdc["setting"]["output"] + "_det.txt", "a") as f:
        # Write time step and iteration
        f.write(
            "Time Step "
            + str(
                mcdc["technique"]["census_idx"]
                + 1
                + int((mcdc["i_cycle"] + 1) / (mcdc["technique"]["updates"] + 1))
            )
        )
        f.write(
            " Update "
            + str((mcdc["i_cycle"] + 1) % (mcdc["technique"]["updates"] + 1))
            + "\n"
        )
        f.write("Max balance equation residual: %16.5e\n" % np.max(res1))
        f.write("Max 1st moment equation residual: %16.5e\n" % np.max(res2))

        # Write the inputs and outputs
        f.write("       x_mid")
        f.write("   Scalar flux 0")
        f.write("     Scalar flux")
        f.write("       Eddington")
        f.write("      Interp VEF")
        f.write("           x")
        f.write("       Current 0")
        f.write("         Current")
        f.write("\n")
        for i in range(Nx):
            f.write(
                "%12.3f%16.5e%16.5e%16.5e%16.5e%12.3f%16.5e%16.5e\n"
                % (
                    x_mid[i],
                    phi_in[i + 1],
                    phi[i],
                    Edd_raw[i],
                    Edd_in[i],
                    x[i],
                    J_in[i],
                    J[i],
                )
            )
        f.write("%72.3f%16.5e%16.5e\n" % (x[Nx], J_in[Nx], J[Nx]))
        f.write("\n")

    return phi, J

# Runga-Kutta Explicit Modified Euler (unfinished)
@njit
def QD1DME(mcdc, phi_in, J_in, Edd_raw, Edd_raw0):
    return phi, J
