import numpy as np

kmax = int(1e3)


def QD():
    global mu, w
    global Nx, x, Nt, t
    global tau, alpha
    global psi, psi_edge
    global phiL, phiR
    global phi, J, phi_edge
    global Edd, Edd_edge
    global CL, CR
    global res1, res2
    global resL, resR
    global iters

    mu, w = np.polynomial.legendre.leggauss(int(M / 2))
    mu = np.concatenate((mu / 2 - 0.5, mu / 2 + 0.5))
    mu = np.reshape(mu, (1, 1, M))
    w = np.concatenate((w / 2, w / 2))
    w = np.reshape(w, (1, 1, M))

    ##initialization
    Nx = len(x) - 1
    x = np.reshape(x, (1, Nx + 1, 1))
    dx = x[:, 1:] - x[:, :-1]
    Nt = len(t) - 1
    t = np.reshape(t, (Nt + 1, 1, 1))
    dt = t[1:] - t[:-1]

    tau = (1 / v / dt + sigmaT) * dx / mu
    alpha = 1 / tau - 1 / (np.exp(tau) - 1)

    psi = np.zeros((Nt + 1, Nx, M))
    psi_edge = np.zeros((Nt + 1, Nx + 1, M))
    # initial condition
    psi[:, :, :] = IC
    psi_edge[:, :, :] = IC
    # boundary conditions
    psi_edge[1:, 0, int(M / 2) :] = leftBC
    psi_edge[1:, -1, : int(M / 2)] = rightBC

    # Computed Quantities of Interest
    phi = np.sum(psi * w, 2)
    phi_edge = np.sum(psi_edge * w, 2)
    J = np.sum(psi_edge * w * mu, 2)
    Edd = (np.sum(psi * w * mu * mu, 2) + delta) / (np.sum(psi * w, 2) + 3 * delta)
    Edd_edge = (np.sum(psi_edge * w * mu * mu, 2) + delta) / (
        np.sum(psi_edge * w, 2) + 3 * delta
    )
    CL = np.reshape(
        np.sum(
            psi_edge[:, 0, : int(M / 2)]
            * w[:, :, : int(M / 2)]
            * mu[:, :, : int(M / 2)],
            2,
        )
        / (np.sum(psi_edge[:, 0, : int(M / 2)] * w[:, :, : int(M / 2)], 2) + delta),
        (Nt + 1),
    )
    CR = np.reshape(
        np.sum(
            psi_edge[:, -1, int(M / 2) :]
            * w[:, :, int(M / 2) :]
            * mu[:, :, int(M / 2) :],
            2,
        )
        / (np.sum(psi_edge[:, -1, int(M / 2) :] * w[:, :, int(M / 2) :], 2) + delta),
        (Nt + 1),
    )
    res1 = np.zeros((Nt, Nx))
    res2 = np.zeros((Nt, Nx + 1))
    resL = np.zeros((Nt,))
    resR = np.zeros((Nt,))
    iters = np.zeros((Nt,))

    JLin = np.sum(
        psi_edge[1, 0, int(M / 2) :] * w[:, :, int(M / 2) :] * mu[:, :, int(M / 2) :], 2
    )
    JRin = np.sum(
        psi_edge[1, -1, : int(M / 2)] * w[:, :, : int(M / 2)] * mu[:, :, : int(M / 2)],
        2,
    )
    phiLin = np.sum(psi_edge[1, 0, int(M / 2) :] * w[:, :, int(M / 2) :], 2)
    phiRin = np.sum(psi_edge[1, -1, : int(M / 2)] * w[:, :, : int(M / 2)], 2)

    # time loop
    for s in range(Nt):
        # initial estimate of eddington tensors
        Edd[s + 1] = np.copy(Edd[s])
        Edd_edge[s + 1] = np.copy(Edd_edge[s])
        CL[s + 1] = np.copy(CL[s])
        CR[s + 1] = np.copy(CR[s])

        dx2 = np.zeros((Nx + 1, 1))
        dx2[0] = dx[:, 0] / 2
        dx2[1:-1] = (dx[:, 1:] + dx[:, :-1]) / 2
        dx2[-1] = dx[:, -1] / 2
        dx2 = np.squeeze(dx2)

        Edd2 = np.zeros((Nx + 2,))
        dt2 = np.squeeze(dt[s])
        for itr in range(kmax):
            if method == 1:
                # LOQD Solver
                phi_old = np.copy(phi[s + 1])
                Edd2[0] = Edd_edge[s + 1, 0]
                Edd2[1:-1] = Edd[s + 1]
                Edd2[-1] = Edd_edge[s + 1, -1]

                # Solve transport problem using QD given eddington tensors
                # build tridiagonal matrix for cell
                a = Edd2[:-1] / dx2 / (1 / v / dt2 + sigmaT)
                b = Edd2[1:] / dx2 / (1 / v / dt2 + sigmaT)
                f = np.zeros((Nx + 2,))
                f[0] = -CL[s + 1]
                f[1:-1] = np.squeeze(dx) * (sigmaT - sigmaS + 1 / v / dt2)
                f[-1] = CR[s + 1]
                d = np.zeros((Nx + 2,))
                d[0] = (
                    JLin
                    - CL[s + 1] * phiLin
                    - J[s, 0] / v / dt2 / (sigmaT + 1 / v / dt2)
                )
                d[1:-1] = np.squeeze(dx) * (q + phi[s] / v / dt2) + (1 / v / dt2) / (
                    sigmaT + 1 / v / dt2
                ) * (J[s, :-1] - J[s, 1:])
                d[-1] = (
                    CR[s + 1] * phiRin
                    - JRin
                    + J[s, -1] / v / dt2 / (sigmaT + 1 / v / dt2)
                )
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
                    dd[i + 1] = (dd[i + 1] - aa[i] * dd[i]) / (
                        bb[i + 1] - aa[i] * cc[i]
                    )
                # last row
                dd[Nx + 1] = (dd[Nx + 1] - aa[Nx] * dd[Nx]) / (
                    bb[Nx + 1] - aa[Nx] * cc[Nx]
                )

                # last row
                phi_edge[s + 1, -1] = dd[Nx + 1]
                phi[s + 1, Nx - 1] = dd[Nx] - cc[Nx] * phi_edge[s + 1, -1]
                # other rows
                for i in range(Nx - 1):
                    phi[s + 1, Nx - i - 2] = (
                        dd[Nx - i - 1] - cc[Nx - i - 1] * phi[s + 1, Nx - i - 1]
                    )
                phi_edge[s + 1, 0] = dd[0] - cc[0] * phi[s + 1, 0]

                # solve for current by multiplication
                for i in range(Nx - 1):
                    J[s + 1, i + 1] = (
                        (
                            Edd2[i + 1] * phi[s + 1, i]
                            - Edd2[i + 2] * phi[s + 1, i + 1]
                            + J[s, i + 1] * dx2[i + 1] / v / dt2
                        )
                        / dx2[i + 1]
                        / (1 / v / dt2 + sigmaT)
                    )
                J[s + 1, 0] = (
                    (
                        Edd2[0] * phi_edge[s + 1, 0]
                        - Edd2[1] * phi[s + 1, 0]
                        + J[s, 0] * dx2[0] / v / dt2
                    )
                    / dx2[0]
                    / (1 / v / dt2 + sigmaT)
                )
                J[s + 1, -1] = (
                    (
                        Edd2[-2] * phi[s + 1, -1]
                        - Edd2[-1] * phi_edge[s + 1, -1]
                        + J[s, -1] * dx2[-1] / v / dt2
                    )
                    / dx2[-1]
                    / (1 / v / dt2 + sigmaT)
                )

                # residuals
                for i in range(Nx):
                    res1[s, i] = (
                        J[s + 1, i + 1]
                        - J[s + 1, i]
                        + (1 / v / dt2 + (sigmaT - sigmaS))
                        * np.squeeze(dx[:, i])
                        * phi[s + 1, i]
                        - np.squeeze(dx[:, i]) / v / dt2 * phi[s, i]
                        - q * np.squeeze(dx[:, i])
                    )

                res2[s, 0] = (
                    Edd2[1] * phi[s + 1, 0]
                    - Edd2[0] * phi_edge[s + 1, 0]
                    + sigmaT * dx2[0] * J[s + 1, 0]
                    + dx2[0] / v / dt2 * (J[s + 1, 0] - J[s, 0])
                )
                res2[s, -1] = (
                    Edd2[-1] * phi_edge[s + 1, -1]
                    - Edd2[-2] * phi[s + 1, -1]
                    + sigmaT * dx2[-1] * J[s + 1, -1]
                    + dx2[-1] / v / dt2 * (J[s + 1, -1] - J[s, -1])
                )
                for i in range(Nx - 1):
                    res2[s, i + 1] = (
                        Edd2[i + 2] * phi[s + 1, i + 1]
                        - Edd2[i + 1] * phi[s + 1, i]
                        + sigmaT * dx2[i + 1] * J[s + 1, i + 1]
                        + dx2[i + 1] / v / dt2 * (J[s + 1, i + 1] - J[s, i + 1])
                    )

                resL[s] = J[s + 1, 0] - CL[s + 1] * (phi_edge[s + 1, 0] - phiLin) - JLin
                resR[s] = (
                    J[s + 1, -1] - CR[s + 1] * (phi_edge[s + 1, -1] - phiRin) - JRin
                )

                phi_edge[s + 1, 1:-1] = (
                    sigmaT * np.squeeze(dx[:, :-1]) * Edd[s + 1, 1:] * phi[s + 1, 1:]
                    + sigmaT * np.squeeze(dx[:, 1:]) * Edd[s + 1, :-1] * phi[s + 1, :-1]
                ) / (
                    Edd_edge[s + 1, 1:-1]
                    * np.squeeze(sigmaT * dx[:, :-1] + sigmaT * dx[:, 1:])
                )

                if any(np.abs(res1[s]) > 1e-4):
                    print("Warning: Balance Equation Residual")
                if any(np.abs(res2[s]) > 1e-4):
                    print("Warning: 1st Moment Equation Residual")

            # Calculate spectral radius

            # Check convergence criteria
            if itr > 0 and np.amax(np.abs(phi[s + 1] - phi_old)) < eps_phi:
                iters[s] = itr + 1
                print(
                    "Time Step "
                    + str(s + 1)
                    + " converged in "
                    + str(itr + 1)
                    + " iterations"
                )
                break

            # Check if maximum iterations
            if itr >= kmax - 1:
                iters[s] = itr + 1
                print("Warning: Maximum iterations reached")
                break

            # Step characteristics 1D transport

            # positive direction
            for i in range(Nx):
                psi_edge[s + 1, i + 1, int(M / 2) :] = (
                    dx[:, i] / v / dt[s] * psi_edge[s, i + 1, int(M / 2) :]
                    + 0.5 * (sigmaS * phi[s + 1, i] + q) * dx[:, i]
                    + (
                        mu[:, :, int(M / 2) :]
                        - (1 / v / dt[s] + sigmaT)
                        * dx[:, i]
                        * alpha[s, i, int(M / 2) :]
                    )
                    * psi_edge[s + 1, i, int(M / 2) :]
                ) / (
                    mu[:, :, int(M / 2) :]
                    + (1 / v / dt[s] + sigmaT)
                    * dx[:, i]
                    * (1 - alpha[s, i, int(M / 2) :])
                )

            # negative direction
            for i in range(Nx):
                psi_edge[s + 1, -i - 2, : int(M / 2)] = (
                    -dx[:, i - 1] / v / dt[s] * psi_edge[s, -i - 2, : int(M / 2)]
                    - 0.5 * (sigmaS * phi[s + 1, -i - 1] + q) * dx[:, -i - 1]
                    + (
                        mu[:, :, : int(M / 2)]
                        + (1 / v / dt[s] + sigmaT)
                        * dx[:, -i - 1]
                        * (1 - alpha[s, -i - 1, : int(M / 2)])
                    )
                    * psi_edge[s + 1, -i - 1, : int(M / 2)]
                ) / (
                    mu[:, :, : int(M / 2)]
                    - (1 / v / dt[s] + sigmaT)
                    * dx[:, -i - 1]
                    * alpha[s, -i - 1, : int(M / 2)]
                )

            # Warnings
            if np.any(psi_edge[s + 1, :, int(M / 2) :] < 0):
                print("Warning: +mu Transport Sweep")
            if np.any(psi_edge[s + 1, :, : int(M / 2)] < 0):
                print("Warning: -mu Transport Sweep")

            psi[s + 1] = (
                alpha[s] * psi_edge[s + 1, :-1] + (1 - alpha[s]) * psi_edge[s + 1, 1:]
            )

            # Computed Quantities
            if method == 0:
                phi_old = np.copy(phi[s + 1])
                phi[s + 1] = np.sum(psi[s + 1] * w, 2)
                J[s + 1] = np.sum(psi_edge[s + 1] * w * mu, 2)
            Edd[s + 1] = (np.sum(psi[s + 1] * w * mu * mu, 2) + delta) / (
                np.sum(psi[s + 1] * w, 2) + 3 * delta
            )
            Edd_edge[s + 1] = (np.sum(psi_edge[s + 1] * w * mu * mu, 2) + delta) / (
                np.sum(psi_edge[s + 1] * w, 2) + 3 * delta
            )
            CL[s + 1] = np.sum(
                psi_edge[s + 1, 0, : int(M / 2)]
                * w[:, :, : int(M / 2)]
                * mu[:, :, : int(M / 2)]
            ) / (
                np.sum(psi_edge[s + 1, 0, : int(M / 2)] * w[:, :, : int(M / 2)]) + delta
            )
            CR[s + 1] = np.sum(
                psi_edge[s + 1, -1, int(M / 2) :]
                * w[:, :, int(M / 2) :]
                * mu[:, :, int(M / 2) :]
            ) / (
                np.sum(psi_edge[s + 1, -1, int(M / 2) :] * w[:, :, int(M / 2) :])
                + delta
            )

            # Warnings
            if any(Edd[s + 1] > 1) or any(Edd[s + 1] <= 0):
                print("Warning: Eddington Tensor")
            if any(Edd_edge[s + 1] > 1) or any(Edd_edge[s + 1] <= 0):
                print("Warning: Eddington Tensor")
            if CL[s + 1] > 1 or CL[s + 1] < -1 or CR[s + 1] > 1 or CR[s + 1] < -1:
                print("Warning: Eddington Tensor")


def xt_write(f, var):
    f.write("      ")
    f.write("   ix   ")
    for i in range(Nx + 1):
        f.write("%12.1f" % (i + 0.5))
    f.write("\n")
    f.write("   it ")
    f.write("  t/x   ")
    for i in range(Nx + 1):
        f.write("%12.2f" % x[:, i])
    f.write("\n")
    for j in range(Nt + 1):
        f.write("%6d" % (j))
        f.write("%8.2f" % t[j])
        for i in range(Nx + 1):
            f.write("%12.4e" % var[j][i])
        f.write("\n")
    f.write("\n")


def xt2_write(f, var):
    f.write("      ")
    f.write("   ix   ")
    for i in range(Nx + 1):
        f.write("%12.1f" % (i + 0.5))
    f.write("\n")
    f.write("   it ")
    f.write("  t/x   ")
    for i in range(Nx + 1):
        f.write("%12.2f" % x[:, i])
    f.write("\n")
    for j in range(Nt):
        f.write("%6d" % (j + 1))
        f.write("%8.2f" % t[j])
        for i in range(Nx + 1):
            f.write("%12.4e" % var[j][i])
        f.write("\n")
    f.write("\n")


def x2t2_write(f, var):
    f.write("      ")
    f.write("   ix   ")
    for i in range(Nx):
        f.write("%12d" % (i + 1))
    f.write("\n")
    f.write("   it ")
    f.write("  t/x   ")
    for i in range(Nx):
        f.write("%12.2f" % x[:, i])
    f.write("\n")
    for j in range(Nt):
        f.write("%6d" % (j + 1))
        f.write("%8.2f" % t[j])
        for i in range(Nx):
            f.write("%12.4e" % var[j][i])
        f.write("\n")
    f.write("\n")


def x2t_write(f, var):
    f.write("      ")
    f.write("   ix   ")
    for i in range(Nx):
        f.write("%12d" % (i + 1))
    f.write("\n")
    f.write("   it ")
    f.write("  t/x   ")
    for i in range(Nx):
        f.write("%12.2f" % x[:, i])
    f.write("\n")
    for j in range(Nt + 1):
        f.write("%6d" % (j))
        f.write("%8.2f" % t[j])
        for i in range(Nx):
            f.write("%12.4e" % var[j][i])
        f.write("\n")
    f.write("\n")


def t_write(f, var):
    for i in range(Nt + 1):
        f.write("%12.4e" % var[i])
    f.write("\n")


def t2_write(f, var):
    for i in range(Nt):
        f.write("%12.4e" % var[i])
    f.write("\n")


def write(outfile="output.txt"):
    f = open(outfile, "w")

    # Overall Run statistics
    f.write("Maximum Balance Equation Residual: " + str(np.amax(np.abs(res1))) + "\n")
    f.write("Maximum 1st Equation Residual: " + str(np.amax(np.abs(res2))) + "\n")
    f.write("\n")
    # Time Dependent Values
    f.write("Iterations\n")
    t2_write(f, iters)
    f.write("Left boundary QD factor\n")
    t_write(f, CL)
    f.write("Right boundary QD factor\n")
    t_write(f, CR)
    f.write("Left boundary QD redisual\n")
    t2_write(f, resL)
    f.write("Right boundary QD redisual\n")
    t2_write(f, resR)
    f.write("\n")

    # Time-Space Variables
    f.write("Scalar Flux\n")
    x2t_write(f, phi)
    f.write("Current\n")
    xt_write(f, J)
    f.write("Edge Eddington Factor\n")
    xt_write(f, Edd_edge)
    f.write("Balance Equation Residual\n")
    x2t2_write(f, res1)
    f.write("1st Moment Equation Residual\n")
    xt2_write(f, res2)
    f.write("\n")

    f.close()


def tswrite(outfolder=""):
    for Nti in range(Nt):
        f = open(outfolder + str(Nti + 1) + ".txt", "w")

        # Write Inputs
        # Quadrature
        f.write("Angular Quadrature\n")
        f.write("  im")
        f.write("      mu_m")
        f.write("       w_m")
        f.write("\n")
        for i in range(M):
            f.write("%4d" % (i + 1))
            f.write(" %11.8f" % mu[0, 0, i])
            f.write(" %11.8f" % w[0, 0, i])
            f.write("\n")

        f.write("\n")
        f.write("  ix")  # 4
        f.write(" Sigma_t")  # 8
        f.write(" Sigma_s")
        f.write("       q")
        f.write("    phi_left")  # 10
        f.write("     phi_ave")
        f.write("      J_left")  # 10
        f.write("     Edd_ave")
        f.write("\n")
        for i in range(Nx):
            f.write("%4d" % (i + 1))
            f.write(" %7.2f" % sigmaT)  # 8
            f.write(" %7.2f" % sigmaS)  # 8
            f.write(" %7.2f" % q)  # 8
            f.write(" %11.4e" % phi_edge[Nti + 1][i])
            f.write(" %11.4e" % phi[Nti + 1][i])
            f.write(" %11.4e" % J[Nti + 1][i])
            f.write(" %11.4e" % Edd[Nti + 1][i])
            f.write("\n")
        f.write("%4d" % (Nx + 1))
        f.write(" %7.2f" % sigmaT)  # 8
        f.write(" %7.2f" % sigmaS)  # 8
        f.write(" %7.2f" % q)  # 8
        f.write(" %11.4e" % phi_edge[Nti + 1][Nx])
        f.write(" n/a        ")
        f.write(" %11.4e" % J[Nti + 1][Nx])
        f.write(" n/a        ")
        f.write("\n")

        # Overall Run statistics
        f.write(
            "Maximum Balance Equation Residual: "
            + str(np.amax(np.abs(res1[Nti])))
            + "\n"
        )
        f.write(
            "Maximum 1st Equation Residual: " + str(np.amax(np.abs(res2[Nti]))) + "\n"
        )
        f.write("\n")

        f.close()
