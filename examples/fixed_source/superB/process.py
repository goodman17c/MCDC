import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import pandas as pd

method = ["8"]
Np = [400, 1000, 4000, 10000]
Np = [1000, 4000, 10000]
updatelist = [0, 1, 2, 3, 10]
updatelist = [1, 2, 3, 10]
ref = ["reference.npz", "reference_40.npz", "reference_80.npz"]
for Npi in Np:
    for fj in range(1, 2):
        for updates in updatelist:
            for methodi in method:
                output = (
                    methodi + "_" + str(2**fj) + "_" + str(updates) + "_" + str(Npi)
                )
                # output="output"

                # =============================================================================
                # Reference solution (SS)
                # =============================================================================

                data = np.load("reference.npz")
                phi_ref = data["phi"]
                phi_t_ref = data["phi_t"]
                phi_x_ref = data["phi_x"]

                with h5py.File(output + ".h5", "r") as f:
                    phi = f["tally/flux/mean"][:]
                    phi_sd = f["tally/flux/sdev"][:]
                    phit = f["tally/flux-t/mean"][:]
                    phit_sd = f["tally/flux-t/sdev"][:]
                    phix = f["tally/flux-x/mean"][:]
                    phix_sd = f["tally/flux-x/sdev"][:]
                    current = f["tally/current/mean"][:]
                    current_sd = f["tally/current/sdev"][:]
                    currentt = f["tally/current-t/mean"][:]
                    currentt_sd = f["tally/current-t/sdev"][:]
                    currentx = f["tally/current-x/mean"][:]
                    currentx_sd = f["tally/current-x/sdev"][:]
                    eddington = f["tally/eddington/mean"][:]
                    eddington_sd = f["tally/eddington/sdev"][:]
                    eddingtont = f["tally/eddington-t/mean"][:]
                    eddingtont_sd = f["tally/eddington-t/sdev"][:]
                    eddingtonx = f["tally/eddington-x/mean"][:]
                    eddingtonx_sd = f["tally/eddington-x/sdev"][:]
                    n = f["tally/n/mean"][:]
                    n_t = f["tally/n-t/mean"][:]
                    n_x = f["tally/n-t/mean"][:]
                    ww = f["ww/center"][:]
                    x = f["tally/grid/x"][:]
                    t = f["tally/grid/t"][:]

                dx = x[1] - x[0]
                x_mid = 0.5 * (x[:-1] + x[1:])
                dt = t[1:] - t[:-1]
                t_mid = 0.5 * (t[:-1] + t[1:])
                K = len(dt)
                J = len(x_mid)

                # Load deterministic iteration data
                phi_det = np.zeros((K, updates + 1, J))
                J_det = np.zeros((K, updates + 1, J + 1))
                Edd_det = np.zeros((K, updates + 1, J))
                with open(output + "_det.txt", "r") as f:
                    for k in range(K):
                        for u in range(updates + 1):
                            if k == 0 and u == 0:
                                continue
                            f.readline()
                            f.readline()
                            f.readline()
                            f.readline()
                            for i in range(J):
                                line = f.readline().split()
                                phi_det[k][u][i] = float(line[2])
                                J_det[k][u][i] = float(line[6])
                                Edd_det[k][u][i] = float(line[3])
                            line = f.readline().split()
                            J_det[k][u][J] = float(line[2])
                            f.readline()

                for k in range(K):
                    phi[k] /= dx * dt[k]
                    phi_sd[k] /= dx * dt[k]
                    phit[k] /= dx
                    phit_sd[k] /= dx
                    phix[k] /= dt[k]
                    phix_sd[k] /= dt[k]
                    current[k] /= dx * dt[k]
                    current_sd[k] /= dx * dt[k]
                    currentt[k] /= dx
                    currentt_sd[k] /= dx
                    currentx[k] /= dt[k]
                    currentx_sd[k] /= dt[k]
                    eddington[k] /= dx * dt[k]
                    eddington_sd[k] /= dx * dt[k]
                    eddingtont[k] /= dx
                    eddingtont_sd[k] /= dx
                    eddingtonx[k] /= dt[k]
                    eddingtonx_sd[k] /= dt[k]

                FOM = phi * phi / phi_sd / phi_sd / np.sum(n) / Npi
                FOM[n == 0] = 0
                newFOM = (
                    phi_ref
                    * phi_ref
                    / (phi - phi_ref)
                    / (phi - phi_ref)
                    / np.sum(n)
                    / Npi
                )
                newFOM[phi_ref == 0] = 0

                # Calculate Integral quatities
                phi_int = np.mean(phi, 1) * 40.2
                n_int = np.sum(n, 1)
                n_t_int = np.sum(n_t, 1)

                max_rel_err = np.zeros(K)
                err_inf = np.zeros(K)
                err_L1 = np.zeros(K)
                err_L2 = np.zeros(K)
                err_2 = np.zeros(K)
                rel_err_inf = np.zeros(K)
                rel_err_L1 = np.zeros(K)
                rel_err_L2 = np.zeros(K)
                rel_err_2 = np.zeros(K)
                stat_err = np.zeros(K)

                # Analysis of Numerical Solutions
                for k in range(K):
                    max_rel_err[k] = np.max(
                        np.abs(1 - np.nan_to_num(phi[k] / phi_ref[k], nan=1))
                    )
                    err_inf[k] = np.max(np.abs(phi[k] - phi_ref[k]))
                    err_L1[k] = np.sum(np.abs(phi[k] - phi_ref[k]) * dx)
                    err_L2[k] = np.sqrt(np.sum(np.power(phi[k] - phi_ref[k], 2) * dx))
                    err_2[k] = np.sqrt(np.sum(np.power(phi[k] - phi_ref[k], 2)))
                    rel_err_inf[k] = err_inf[k] / np.max(np.abs(phi_ref[k]))
                    rel_err_L1[k] = err_L1[k] / np.sum(np.abs(phi_ref[k]) * dx)
                    rel_err_L2[k] = err_L2[k] / np.sqrt(
                        np.sum(np.power(phi_ref[k], 2) * dx)
                    )
                    rel_err_2[k] = err_2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2)))
                    stat_err[k] = np.sqrt(
                        np.sum(np.power(phi_sd[k], 2)) / np.sum(np.power(phi[k], 2))
                    )

                # Write data to large matrix
                # A column for each parameter on a single time step 4 MC scalar fluxs, ww, 3 currents, 3 eddington
                columnlist = [
                    "x_mid",
                    "ww",
                    "phi",
                    "phi_sd",
                    "phi_ref",
                    "phi_t",
                    "phi_t_sd",
                    "phi_t_ref",
                    "x",
                    "phi_x",
                    "phi_x_sd",
                    "phi_x_ref",
                ]
                columnlistJ = [
                    "x_mid",
                    "current",
                    "current_sd",
                    "current_t",
                    "current_t_sd",
                    "x",
                    "current_x",
                    "current_x_sd",
                ]
                columnlistEdd = [
                    "x_mid",
                    "Eddington",
                    "Eddington_sd",
                    "Eddington_t",
                    "Eddington_t_sd",
                    "x",
                    "Eddington_x",
                    "Eddington_x_sd",
                ]
                print(np.shape(phi[0]))
                data = np.full((K * (J + 2), 12), np.nan)
                dataJ = np.full((K * (J + 2), 8), np.nan)
                dataEdd = np.full((K * (J + 2), 8), np.nan)
                for k in range(K):
                    data[k * (J + 2), 0] = t_mid[k]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 0] = x_mid[i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 1] = ww[k][i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 2] = phi[k][i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 3] = phi_sd[k][i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 4] = phi_ref[k][i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 5] = phit[k + 1][i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 6] = phit_sd[k + 1][i]
                    for i in range(J):
                        data[k * (J + 2) + i + 1, 7] = phi_t_ref[k][i]
                    for i in range(J + 1):
                        data[k * (J + 2) + i + 1, 8] = x[i]
                    for i in range(J + 1):
                        data[k * (J + 2) + i + 1, 9] = phix[k][i]
                    for i in range(J + 1):
                        data[k * (J + 2) + i + 1, 10] = phix_sd[k][i]
                    for i in range(J + 1):
                        data[k * (J + 2) + i + 1, 11] = phi_x_ref[k][i]

                    dataJ[k * (J + 2), 0] = t_mid[k]
                    for i in range(J):
                        dataJ[k * (J + 2) + i + 1, 0] = x_mid[i]
                    for i in range(J):
                        dataJ[k * (J + 2) + i + 1, 1] = current[k][i][0]
                    for i in range(J):
                        dataJ[k * (J + 2) + i + 1, 2] = current_sd[k][i][0]
                    for i in range(J):
                        dataJ[k * (J + 2) + i + 1, 3] = currentt[k + 1][i][0]
                    for i in range(J):
                        dataJ[k * (J + 2) + i + 1, 4] = currentt_sd[k + 1][i][0]
                    for i in range(J + 1):
                        dataJ[k * (J + 2) + i + 1, 5] = x[i]
                    for i in range(J + 1):
                        dataJ[k * (J + 2) + i + 1, 6] = currentx[k][i][0]
                    for i in range(J + 1):
                        dataJ[k * (J + 2) + i + 1, 7] = currentx_sd[k][i][0]

                    dataEdd[k * (J + 2), 0] = t_mid[k]
                    for i in range(J):
                        dataEdd[k * (J + 2) + i + 1, 0] = x_mid[i]
                    for i in range(J):
                        dataEdd[k * (J + 2) + i + 1, 1] = eddington[k][i][0] / phi[k][i]
                    for i in range(J):
                        dataEdd[k * (J + 2) + i + 1, 2] = (
                            eddington_sd[k][i][0] / phi[k][i]
                        )
                    for i in range(J):
                        dataEdd[k * (J + 2) + i + 1, 3] = (
                            eddingtont[k + 1][i][0] / phit[k + 1][i]
                        )
                    for i in range(J):
                        dataEdd[k * (J + 2) + i + 1, 4] = (
                            eddingtont_sd[k + 1][i][0] / phit[k + 1][i]
                        )
                    for i in range(J + 1):
                        dataEdd[k * (J + 2) + i + 1, 5] = x[i]
                    for i in range(J + 1):
                        dataEdd[k * (J + 2) + i + 1, 6] = (
                            eddingtonx[k][i][0] / phix[k][i]
                        )
                    for i in range(J + 1):
                        dataEdd[k * (J + 2) + i + 1, 7] = (
                            eddingtonx_sd[k][i][0] / phix[k][i]
                        )

                # =============================================================================
                # Print results
                # =============================================================================

                with pd.ExcelWriter(output + ".xlsx") as writer:
                    # Complete data in time and space
                    df = pd.DataFrame(data, columns=columnlist)
                    df.to_excel(writer, sheet_name="Scalar Flux")
                    df = pd.DataFrame(dataJ, columns=columnlistJ)
                    df.to_excel(writer, sheet_name="Current")
                    df = pd.DataFrame(dataEdd, columns=columnlistEdd)
                    df.to_excel(writer, sheet_name="Eddington")

                    # Data normed over space

                    # Copy deterministic statistic from file for each update (predictor or corrector)

                    # Run Overview statistics and flags

                # =============================================================================
                # Animate results
                # =============================================================================

                # Integral Flux
                # plt.plot(t_mid,phi_int,label="MC")
                # plt.plot(t_mid,n_int,label="n")
                # plt.grid()
                # plt.xlabel(r'$t$')
                # plt.ylabel(r'Integral Flux')
                # plt.show()

                # Flux - average
                fig, ax = plt.subplots()
                ax.grid()
                ax.set_ylim([1e-12, 1e2])
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"Flux")
                (line1,) = ax.semilogy([], [], "-b", label="MC")
                (line2,) = ax.semilogy([], [], "--r", label="Ref.")
                (line3,) = ax.semilogy([], [], "-.g", label="FOM")
                (line4,) = ax.semilogy([], [], ":m", label="n")
                text = ax.text(0.02, 0.9, "", transform=ax.transAxes)
                ax.legend()

                def animate(k):
                    line1.set_data(x_mid, phi[k, :])
                    ax.collections.clear()
                    ax.fill_between(
                        x_mid,
                        phi[k, :] - phi_sd[k, :],
                        phi[k, :] + phi_sd[k, :],
                        alpha=0.2,
                        color="b",
                    )
                    line2.set_data(x_mid, phi_ref[k, :])
                    line3.set_data(x_mid, newFOM[k, :])
                    line4.set_data(x_mid, n[k, :])
                    text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[k], t[k + 1]))
                    return line1, line2, line3, text

                simulation = animation.FuncAnimation(fig, animate, frames=K)
                writervideo = animation.FFMpegWriter(fps=6)
                simulation.save(output + ".mp4", writer=writervideo)
                # plt.show()
                plt.close()
