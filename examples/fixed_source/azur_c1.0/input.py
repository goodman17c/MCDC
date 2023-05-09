import numpy as np

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
mcdc.tally(
    scores=["flux", "current-x", "eddington", "n", "n_t"],
    x=np.linspace(-20.1, 20.1, 202),
    t=t,
)

# Setting
mcdc.setting(
    N_particle=1e2,
    #             rng_seed=123,
    time_boundary=20.1,
    active_bank_buff=2e7,
    census_bank_buff=1e4,
)


data = np.load("reference.npz")
phi_ref = data["phi"]
phi_t_ref = data["phi_t"]
# phi_t_ref[1:]=phi_t_ref[:-1]
phi_t_ref[:, :] = 0
phi_t_ref[:, 100] = 1
for k in range(20):
    phi_ref[k] = phi_ref[k] / np.max(phi_ref[k])
#    phi_t_ref[k] = phi_t_ref[k]/np.max(phi_t_ref[k])
# phi_ref[1:]=phi_ref[:-1] #Use weight windows from previous time step
mcdc.auto_ww(method="cooper")
mcdc.weight_window(
    x=np.linspace(-20.1, 20.1, 202),
    t=t,
    rho=4.0,
    wwtype="isotropic",
    window=phi_ref[0:5],
)

mcdc.census(t=t[1:], pct="combing")
# mcdc.census(t=t[1:])

# Run
mcdc.run()
