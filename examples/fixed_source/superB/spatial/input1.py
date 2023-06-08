Nx = [40, 1000]
ref = ["reference_40.npz", "reference_1000.npz"]
Nplist = [400, 1000, 4000, 10000, 40000]
for Np in Nplist:
    for i in range(2):
        for j in range(1):
            import numpy as np
            import mcdc

            # =============================================================================
            # Set model
            # =============================================================================
            # Infinite medium with isotropic plane surface at the center
            # Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
            # Effective scattering ratio c = 1.0

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
            t = np.linspace(0.0, 20.0, 21)
            t[0] = -0.0001
            mcdc.tally(
                scores=["flux", "n", "n_t"], x=np.linspace(-20-20/Nx[i], 20+20/Nx[i], Nx[i]+2), t=t
            )

            # Setting
            mcdc.setting(
                N_particle=Np,
                time_boundary=20.1,
                active_bank_buff=2e7,
                census_bank_buff=1e3,
                output="1_" + str(Nx[i]) + "_" + str(Np),
            )

            mcdc.implicit_capture()
            mcdc.census(t=t[1:], pct="combing")

            # Run
            mcdc.run()
