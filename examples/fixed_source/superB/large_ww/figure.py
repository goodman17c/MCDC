import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

ref = ["reference.npz", "reference_40.npz", "reference_80.npz"]
label = [
    "0",
    "Implicit Capture",
    "Integral WW",
    "Continuous WW",
    "End of TS WW",
    "Average of TS WW",
    "Beginning of TS WW",
    "Hybrid QD WW",
    "8",
    "Hybrid QD WW",
]


def add_to_plot(i, j, k, l):
    method = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    width = ["", "_1", "_2", "_4", "_8","_1000","_10000","_1000000000"]
    Nt = ["_20", "_40", "_80"]
    # =============================================================================
    # Reference solution (SS)
    # =============================================================================

    # Load grids
    data = np.load(ref[k])
    phi_t_ref = data["phi_t"]
    phi_ref = data["phi"]
    x = data["x"]
    t = data["t"]

    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dt = t[1:] - t[:-1]
    t_mid = 0.5 * (t[:-1] + t[1:])
    K = len(dt)
    J = len(x_mid)
    file = method[i] + width[j] + Nt[k] + "_" + str(l) + ".h5"
    with h5py.File(file, "r") as f:
        phi = f["tally/flux/mean"][:]
        phi_sd = f["tally/flux/sdev"][:]
        n = f["tally/n/mean"][:]
        n_t = f["tally/n-t/mean"][:]
        runtime_total = f["runtime_total"][:]
        for k in range(K):
            phi[k] /= dx * dt[k]
            phi_sd[k] /= dx * dt[k]

        FOM = phi * phi / phi_sd / phi_sd / np.sum(n) / l
        FOMt = phi * phi / phi_sd / phi_sd / runtime_total
        FOM[n == 0] = 0
        FOMt[n == 0] = 0
        phi_int = np.sum(n, 1)
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
        missing_domain = np.zeros(K)

        FOM_L2 = np.zeros(K)
        FOMt_L2 = np.zeros(K)

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
            rel_err_L2[k] = err_L2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2) * dx))
            rel_err_2[k] = err_2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2)))
            stat_err[k] = np.sqrt(
                np.sum(np.power(phi_sd[k], 2)) / np.sum(np.power(phi[k], 2))
            )
            missing_domain[k] = 1 - np.count_nonzero(phi[k] > 0) / np.count_nonzero(
                phi_ref[k] > 0
            )

            FOM_L2[k] = np.sqrt(np.sum(np.power(FOM[k], 2) * dx))
            FOMt_L2[k] = np.sqrt(np.sum(np.power(FOMt[k], 2) * dx))

        plt.semilogy(x_mid,phi[-1],label=file)
        #plt.semilogy(t_mid, FOM_L2, label=label[i])
        # plt.semilogy(t[1:],missing_domain,label=label[i])
        #plt.semilogy(t_mid,rel_err_L2,label=label[i])
        # plt.semilogy(t_mid,stat_err,label=file)
        # plt.semilogy(t[1:],n_t_int[1:],label=file)


# =============================================================================
# Animate results
# =============================================================================

# for i in range(4):
# 	for j in range(3):
# add_to_plot(1,i+1,j)
# add_to_plot(2,i+1,j)
# add_to_plot(3,i+1,j)
add_to_plot(1, 0, 0, 10000)
#add_to_plot(2, 3, 0, 10000)
add_to_plot(3, 3, 0, 10000)
add_to_plot(3, 5, 0, 10000)
add_to_plot(3, 6, 0, 10000)
add_to_plot(3, 7, 0, 10000)
#add_to_plot(4, 2, 0, 10000)
#add_to_plot(5, 2, 0, 10000)
#add_to_plot(6, 2, 0, 10000)
# add_to_plot(7,2,0,40000)
# add_to_plot(9,2,0,40000)
plt.grid()
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"Ideal Figure of Merit")
plt.show()
