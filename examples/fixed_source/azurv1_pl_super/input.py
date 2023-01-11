import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Set materials
m = mcdc.material(capture=np.array([1.0/3.0]), scatter=np.array([[1.0/3.0]]),
                  fission=np.array([1.0/3.0]), nu_p=np.array([2.3]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic pulse at x=t=0

mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average, cell-edge, and time-edge scalar fluxes
mcdc.tally(scores=['flux','n'],
           x=np.linspace(-20.5, 20.5, 202),
           t=np.linspace(-0.0001, 20.0, 21))

# Setting
mcdc.setting(N_particle=1E3,
             rng_seed=123,
             time_boundary=20.1,
             active_bank_buff=2E7,
             census_bank_buff=1E3)

mcdc.implicit_capture()

data = np.load('reference.npz')
phi_t_ref = data['phi_t']
phi_ref = data['phi']
phi_ref[1:]=phi_ref[:-1] #Use weight windows from previous time step
t=np.linspace(0.0, 20.0, 21)
t[0] = -1.0
mcdc.weight_window(
           x=np.linspace(-20.5, 20.5, 202), 
           t=t,
           rho=1.0,
           wwtype='isotropic',
           window=phi_ref)

mcdc.census(t=t, pct="combing")

# Run
mcdc.run()
