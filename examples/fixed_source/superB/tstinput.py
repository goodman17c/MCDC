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
t=np.linspace(0.0, 20.0, 21)
t[0] = -0.0001
t=t[0:3]
mcdc.tally(scores=['flux','n','n_t','current_x','eddington'],
					 x=np.linspace(-20.1, 20.1, 202),
					 t=t)

# Setting
mcdc.setting(N_particle=1E3,
						 time_boundary=20.1,
						 active_bank_buff=2E5,
						 census_bank_buff=1E4,
						 output="9_1_20")


phi_ref=np.zeros([2, 201])
phi_ref[0,100]=1
mcdc.weight_window(
					 x=np.linspace(-20.1, 20.1, 202), 
					 t=t,
					 rho=1.0,
					 wwtype='isotropic',
					 window=phi_ref)
mcdc.implicit_capture()
mcdc.auto_ww(method='cooper')
mcdc.census(t=t[1:], pct="combing")

# Run
mcdc.run()
