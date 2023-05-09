import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.05]]))

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=5.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# incoming flux from left boundary

mcdc.source(point=[0.0, 0.0, 0.0], time=[0.0, 1.0], isotropic=True, prob=1e2 * 2 * 1)

mcdc.source(x=[0.0, 5.0], isotropic=True, prob=1e-3 * 5 * 2)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average, cell-edge, and time-edge scalar fluxes
t = np.linspace(0.0, 1.0, 51)
t[0] = -0.0001
mcdc.tally(
    scores=["flux", "current-x", "eddington", "n", "n_t"],
    x=np.linspace(0.0, 5.0, 101),
    t=t,
)

# Setting
mcdc.setting(
    N_particle=1e4,
    #             rng_seed=123,
    time_boundary=1.01,
    active_bank_buff=2e7,
    census_bank_buff=1e4,
)

# Run
mcdc.run()
