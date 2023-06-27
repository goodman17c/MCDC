import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

method = ["8", "11", "18", "21"]
Nt = [20, 40, 80]
Np = [400, 1000, 4000, 10000, 40000]
ref = ["reference.npz", "reference_40.npz", "reference_80.npz"]
for Nti in range(1):
    for fj in range(4):
        for Npi in Np:
            for methodi in method:
                output = (
                    methodi + "_" + str(2**fj) + "_" + str(Nt[Nti]) + "_" + str(Npi)
                )
                # output="output"

                # =============================================================================
                # Reference solution (SS)
                # =============================================================================

                data = np.load(ref[Nti])
                # phi_t_ref = data['phi_t']
                phi_ref = data["phi"]

                with h5py.File(output + ".h5", "r") as f:
                    phi = f["tally/flux/mean"][:]
                    phi_sd = f["tally/flux/sdev"][:]
                    n = f["tally/n/mean"][:]
                    n_t = f["tally/n-t/mean"][:]
                    ww = f["ww/center"][:]
                    x = f["tally/grid/x"][:]
                    t = f["tally/grid/t"][:]

                dx = x[1] - x[0]
                x_mid = 0.5 * (x[:-1] + x[1:])
                dt = t[1:] - t[:-1]
                t_mid = 0.5 * (t[:-1] + t[1:])
                K = len(dt)
                J = len(x_mid)
                for k in range(K):
                    phi[k] /= dx * dt[k]
                    phi_sd[k] /= dx * dt[k]

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

                # =============================================================================
                # Print results
                # =============================================================================

                def print_var(outfile, var):
                    outfile.write("      ")
                    outfile.write("   ix   ")
                    for i in range(J):
                        outfile.write("%12d" % (i + 1))
                    outfile.write("\n")
                    outfile.write("   it ")
                    outfile.write("  t/x   ")
                    for i in range(J):
                        outfile.write("%12.2f" % x_mid[i])
                    outfile.write("\n")
                    for j in range(K):
                        outfile.write("%6d" % (j + 1))
                        outfile.write("%8.2f" % t_mid[j])
                        for i in range(J):
                            outfile.write("%12.4e" % var[j][i])
                        outfile.write("\n")
                    outfile.write("\n")

                def print_int(outfile, var):
                    for i in range(K):
                        outfile.write("%12.4e" % var[i])
                    outfile.write("\n")

                with open(output + ".txt", "w") as outfile:
                    # Integral Quantities
                    outfile.write("\n       it   ")
                    for i in range(K):
                        outfile.write("%12d" % (i + 1))
                    outfile.write("\n")
                    outfile.write("       t    ")
                    print_int(outfile, t_mid)
                    outfile.write("integral phi")
                    print_int(outfile, phi_int)
                    outfile.write("integral n  ")
                    print_int(outfile, n_int)
                    outfile.write("integral n_t")
                    print_int(outfile, n_int)
                    outfile.write("Max Rel Err ")
                    print_int(outfile, max_rel_err)
                    outfile.write("Err Inf Norm")
                    print_int(outfile, err_inf)
                    outfile.write("Err L1 Norm ")
                    print_int(outfile, err_L1)
                    outfile.write("Err L2 Norm ")
                    print_int(outfile, err_L2)
                    outfile.write("Err 2 Norm  ")
                    print_int(outfile, err_2)
                    outfile.write("Rel Err Inf ")
                    print_int(outfile, rel_err_inf)
                    outfile.write("Rel Err L1  ")
                    print_int(outfile, rel_err_L1)
                    outfile.write("Rel Err L2  ")
                    print_int(outfile, rel_err_L2)
                    outfile.write("Rel Err 2   ")
                    print_int(outfile, rel_err_2)
                    outfile.write("Stat Err    ")
                    print_int(outfile, stat_err)

                    # Space and Time data
                    outfile.write("\n")
                    outfile.write("phi\n")
                    print_var(outfile, phi)
                    outfile.write("phi_reference\n")
                    print_var(outfile, phi_ref)
                    outfile.write("phi_sd\n")
                    print_var(outfile, phi_sd)
                    outfile.write("n\n")
                    print_var(outfile, n)
                    outfile.write("FOM\n")
                    print_var(outfile, FOM)
                    outfile.write("n_t\n")
                    print_var(outfile, n_t)
                    outfile.write("weight window\n")
                    print_var(outfile, ww)

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
