import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# =============================================================================
# Reference solution (SS)
# =============================================================================

# Load grids
with h5py.File("WW.h5", "r") as f:
    x = f["tally/grid/x"][:]
    t = f["tally/grid/t"][:]

dx = x[1] - x[0]
x_mid = 0.5 * (x[:-1] + x[1:])
dt = t[1:] - t[:-1]
t_mid = 0.5 * (t[:-1] + t[1:])
K = len(dt)
J = len(x_mid)

data = np.load("reference.npz")
# phi_t_ref = data['phi_t']
phi_ref = data["phi"]

with h5py.File("WW.h5", "r") as f:
    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    n = f["tally/n/mean"][:]
    n_sd = f["tally/n/sdev"][:]
for k in range(K):
    phi[k] /= dx * dt[k]
    phi_sd[k] /= dx * dt[k]

FOM = phi * phi / phi_sd / phi_sd / np.sum(n)
FOM[n == 0] = 0

# Analysis of Numerical Solutions
for k in range(K):
    max_rel_err[k] = np.max(np.abs(1 - np.nan_to_num(phi[k] / phi_ref[k], nan=1)))
    err_inf[k] = np.max(np.abs(phi[k] - phi_ref[k]))
    err_L1[k] = np.sum(np.abs(phi[k] - phi_ref[k]) * dx)
    err_L2[k] = np.sqrt(np.sum(np.power(phi[k] - phi_ref[k], 2) * dx))
    err_2[k] = np.sqrt(np.sum(np.power(phi[k] - phi_ref[k], 2)))
    rel_err_inf[k] = err_inf[k] / np.max(np.abs(phi_ref[k]))
    rel_err_L1[k] = err_L1[k] / np.sum(np.abs(phi_ref[k]) * dx)
    rel_err_L2[k] = err_L2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2) * dx))
    rel_err_2[k] = err_2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2)))

with h5py.File("WWPrev.h5", "r") as f:
    phi_2 = f["tally/flux/mean"][:]
    phi_sd_2 = f["tally/flux/sdev"][:]
    n_2 = f["tally/n/mean"][:]
    n_sd_2 = f["tally/n/sdev"][:]
for k in range(K):
    phi_2[k] /= dx * dt[k]
    phi_sd_2[k] /= dx * dt[k]

FOM_2 = phi_2 * phi_2 / phi_sd_2 / phi_sd_2 / np.sum(n_2)
FOM_2[n == 0] = 0

# Analysis of Numerical Solutions
for k in range(K):
    max_rel_err_2[k] = np.max(np.abs(1 - np.nan_to_num(phi_2[k] / phi_ref[k], nan=1)))
    err_inf_2[k] = np.max(np.abs(phi_2[k] - phi_ref[k]))
    err_L1_2[k] = np.sum(np.abs(phi_2[k] - phi_ref[k]) * dx)
    err_L2_2[k] = np.sqrt(np.sum(np.power(phi_2[k] - phi_ref[k], 2) * dx))
    err_2_2[k] = np.sqrt(np.sum(np.power(phi_2[k] - phi_ref[k], 2)))
    rel_err_inf_2[k] = err_inf_2[k] / np.max(np.abs(phi_ref[k]))
    rel_err_L1_2[k] = err_L1_2[k] / np.sum(np.abs(phi_ref[k]) * dx)
    rel_err_L2_2[k] = err_L2_2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2) * dx))
    rel_err_2_2[k] = err_2_2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2)))

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
