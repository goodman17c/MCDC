import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================
v = 1.383
pi = math.acos(-1)
# Results
with h5py.File("output.h5", "r") as f:
    phi = f["tally/flux/mean"][:]
    phi_octant = f["tally/octant-flux/mean"][:]

phi = np.repeat(np.expand_dims(phi, 3), 8, 3)

kfact = -1.0

ww = phi
ww *= np.power(phi_octant / phi, kfact)
ww /= np.max(ww)  # used a fixed normalization constant

np.savez("ww.npz", phi=ww)
