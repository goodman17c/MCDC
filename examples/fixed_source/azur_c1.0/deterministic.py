import numpy as np
import math
import matplotlib.pyplot as plt
import h5py

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.0

# Set materials
m = mcdc.material(capture=np.array([0.0]), scatter=np.array([[1.0]]))

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic pulse at x=t=0

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average, cell-edge, and time-edge scalar fluxes
t = np.linspace(0.0, 5.0, 6)
t[0] = -0.0001
mcdc.weight_window(x=np.linspace(-20.1, 20.1, 202), t=t, window=np.zeros([5, 201]))

phi = np.zeros([1, 202])
phi[0, 101] = 1
J = np.zeros([1, 201])
Edd = np.ones([1, 202]) / 3

print(phi)
plt.plot(x, phi)
plt.plot(x_mid, phi_ref[0])
plt.show()
