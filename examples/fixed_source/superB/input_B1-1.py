ref = ["reference.npz", "reference_40.npz", "reference_80.npz"]
Nplist = [400, 1000, 4000, 10000]
updateslist = [0, 1, 2, 3, 10]
for Np in Nplist:
    for updates in updateslist:
        for j in range(1, 2):
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
            t = np.linspace(0.0, 20.0, 21)
            t[0] = -0.0001
            mcdc.tally(
                scores=[
                    "n",
                    "n_t",
                    "n_x",
                    "flux",
                    "flux_t",
                    "flux_x",
                    "current",
                    "current_t",
                    "current_x",
                    "eddington",
                    "eddington_t",
                    "eddington_x",
                ],
                x=np.linspace(-20.1, 20.1, 202),
                t=t,
            )

            # Setting
            mcdc.setting(
                save_input_deck=True,
                N_particle=Np,
                time_boundary=20.1,
                active_bank_buff=2e6,
                census_bank_buff=1e2,
                output="B1-1_" + str(2**j) + "_" + str(updates) + "_" + str(Np),
            )

            phi_ref = np.zeros([20, 201])
            phi_ref[0, 100] = 1
            mcdc.weight_window(
                x=np.linspace(-20.1, 20.1, 202),
                t=t,
                width=1.0 * (2**j),
                window=phi_ref,
            )
            mcdc.implicit_capture()
            mcdc.auto_weight_window(
                method="semi-implicit loqd half",
                updates=updates,
            )
            mcdc.census(t=t[1:], pct="combing")

            # Run
            mcdc.run()
