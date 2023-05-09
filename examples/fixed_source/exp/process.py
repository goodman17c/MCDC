import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

# =============================================================================
# Process or Plot results
# =============================================================================
pi = math.acos(-1)
# Results
with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dx = x[1:] - x[:-1]
    Nx = len(x_mid)

    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    n = f["tally/n/mean"][:]

phi /= dx

plt.plot(x_mid, phi)
plt.plot(x_mid, n)
plt.show()
