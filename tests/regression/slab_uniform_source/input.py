import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different materials

# Set materials
m1 = mcdc.material(capture=np.array([1.0]))
m2 = mcdc.material(capture=np.array([1.5]))
m3 = mcdc.material(capture=np.array([2.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=2.0)
s3 = mcdc.surface('plane-x', x=4.0)
s4 = mcdc.surface('plane-x', x=6.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m2)
mcdc.cell([+s2, -s3], m3)
mcdc.cell([+s3, -s4], m1)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the medium

mcdc.source(x=[0.0, 6.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux', 'current', 'flux-x', 'current-x'], 
           x=np.linspace(0.0, 6.0, 61))

# Setting
mcdc.setting(N_hist=3E3)

# Run
mcdc.run()