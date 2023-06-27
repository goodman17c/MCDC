import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Set materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

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
mcdc.tally(
    scores=["n", "n-t", "flux", "flux-t", "flux-x", "current-t", "eddington-t"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)

# Setting
mcdc.setting(
    N_particle=1e3,
    active_bank_buff=2e6,
    census_bank_buff=1e3,
)

# Technique
t = np.linspace(0, 20.0, 21)
t[0] = -1e-8
data = np.load("reference.npz")
phi_ref = data["phi_t"]
for k in range(20):
    phi_ref[k] = phi_ref[k] / np.max(phi_ref[k])
phi_ref = np.zeros_like(phi_ref)
mcdc.weight_window(
    x=np.linspace(-20.5, 20.5, 202),
    t=t,
    width=2.0,
    window=phi_ref,
)

mcdc.census(t=t[1:], pct="combing", closeout=True)
mcdc.auto_weight_window(method="semi-implicit loqd half")

# Run
mcdc.run()
