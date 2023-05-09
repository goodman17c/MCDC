import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# =============================================================================
# Reference solution (SS)
# =============================================================================

# Load grids
with h5py.File("output.h5", "r") as f:
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

with h5py.File("output.h5", "r") as f:
    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    n = f["tally/n/mean"][:]
    n_t = f["tally/n-t/mean"][:]
    Edd = f["tally/eddington-x/mean"][:]
for k in range(K):
    phi[k] /= dx * dt[k]
    phi_sd[k] /= dx * dt[k]

FOM = phi * phi / phi_sd / phi_sd / np.sum(n)
FOM[n == 0] = 0

# Average weight of particles in cell
w_avg = phi / n
w_avg[n == 0] = 0

# Calculate Integral quatities
phi_int = np.mean(phi, 1) * 40.2
n_int = np.sum(n, 1)
n_t_int = np.sum(n_t, 1)

plt.plot(x, Edd[0, :, 0])
plt.show()
