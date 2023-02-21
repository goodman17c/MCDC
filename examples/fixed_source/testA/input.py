Nt= [20, 40, 80]
ref=["reference.npz", "reference_40.npz", "reference_80.npz"]
for i in range(2):
	for j in range(4):

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
		t=np.linspace(0.0, 20.0, Nt[i]+1)
		t[0] = -0.0001
		mcdc.tally(scores=['flux','n','n_t'],
							 x=np.linspace(-20.1, 20.1, 202),
							 t=t)

		# Setting
		mcdc.setting(N_particle=1E3,
		#             rng_seed=123,
								 time_boundary=20.1,
								 active_bank_buff=2E7,
								 census_bank_buff=1E3,
								 output="1d_"+str(2**j)+"_"+str(Nt[i])
	)


		data = np.load(ref[i+1])
		#phi_ref = data['phi']
		phi_t_ref = data['phi_t']
		#phi_t_ref = phi_t_ref[0::2]
		phi_ref=np.zeros([Nt[i], 201])
		phi_ref[1:]=phi_t_ref[:-2:2]
		phi_ref[0,100]=1
		for k in range(Nt[i]):
				phi_ref[k] = phi_ref[k]/np.max(phi_ref[k])
		#		phi_t_ref[k] = phi_t_ref[k]/np.max(phi_t_ref[k])
		#phi_ref[1:]=phi_ref[:-1] #Use weight windows from previous time step
		mcdc.weight_window(
							 x=np.linspace(-20.1, 20.1, 202), 
							 t=t,
							 rho=1.0*(2**j),
							 wwtype='isotropic',
							 window=phi_ref)

		mcdc.census(t=t[1:], pct="combing")
		#mcdc.census(t=t[1:])


		# Run
		mcdc.run()
