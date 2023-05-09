import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================
v = 1.383
pi = math.acos(-1)
# Results
with h5py.File("quartz_python_921600000.h5", "r") as f:
    phi = f["tally/flux/mean"][:]

ww = phi
ww /= np.max(ww[0])  # used a fixed normalization constant

np.savez("ww.npz", ww=ww)
