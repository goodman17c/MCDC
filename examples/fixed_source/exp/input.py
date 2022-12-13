import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.material(capture=np.array([np.log(2)]))

# Set surfaces
sx1 = mcdc.surface('plane-x', x=-0.01, bc="vacuum")
sx2 = mcdc.surface('plane-x', x=3.01, bc="vacuum")

# Set cells
mcdc.cell([+sx1, -sx2], m)

# =============================================================================
# Set source
# =============================================================================

#source
mcdc.source(x=0.0, direction=[1.0,0.0,0.0])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux','n'],
           x=np.linspace(-0.01, 3.0, 302)) 

# Setting
mcdc.setting(N_particle=1E3,
             active_bank_buff=1E4)

# Technique
#mcdc.implicit_capture()

mcdc.weight_window(x=[-1.0,1.0,2.0,4.0],
                   rho=1.0,
                   wwtype='isotropic',
                   window=[1.0,0.5,0.0])

mcdc.run()
