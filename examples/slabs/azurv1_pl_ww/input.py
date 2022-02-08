import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set materials
# =============================================================================

M = mcdc.Material(capture=np.array([1.0/3.0]),
                  scatter=np.array([[1.0/3.0]]),
                  fission=np.array([1.0/3.0]), 
                  nu_p=np.array([2.3]))

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(-1E10, "reflective")
S1 = mcdc.SurfacePlaneX(1E10, "reflective")

# Set cells
C = mcdc.Cell([+S0, -S1], M)
cells = [C]

# =============================================================================
# Set source
# =============================================================================

position = mcdc.DistPoint(mcdc.DistDelta(0.0), mcdc.DistDelta(0.0), 
                          mcdc.DistDelta(0.0))

direction = mcdc.DistPointIsotropic()

time = mcdc.DistDelta(0.0)

Src = mcdc.SourceSimple(position=position, direction=direction, time=time)

sources = [Src]

# =============================================================================
# Set filters and tallies
# =============================================================================

# Load grids
with np.load('azurv1_pl.npz') as f:
    time_filter = mcdc.FilterTime(f['t'])
    spatial_filter = mcdc.FilterPlaneX(f['x'])

T = mcdc.Tally('tally', scores=['flux', 'flux-edge', 'flux-face'],
               spatial_filter=spatial_filter,
               time_filter=time_filter)

tallies = [T]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(cells=cells, sources=sources, tallies=tallies, 
                           N_hist=1E4)

# Set population control and census
simulator.set_pct(census_time=np.array([20.0]))

# Load reference solution for weight window
with np.load('azurv1_pl.npz') as f:
    x = f['x']
    t = f['t']
    window = f['phi']
simulator.set_weight_window(x=x, t=t, window=window)

# Run
simulator.run()