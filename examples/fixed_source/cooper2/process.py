import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================
v=1.383
pi=math.acos(-1)
# Results
with h5py.File('output.h5', 'r') as f:
	x     = f['tally/grid/x'][:]
	x_mid = 0.5*(x[:-1]+x[1:])
	y     = f['tally/grid/y'][:]
	y_mid = 0.5*(y[:-1]+y[1:])
	Nx = len(x_mid)
	Ny = len(y_mid)

	cf = 5*5

	phi    = f['tally/flux/mean'][:]*4*cf
	phi_sd = f['tally/flux/sdev'][:]*4*cf
	n      = f['tally/n/mean'][:]

relvar=phi*phi/phi_sd/phi_sd
FOM=1/relvar/np.sum(n)
FOM[n==0]=0
def print_var(outfile, var):
			outfile.write(' iy/ix')
			for i in range(Nx):
					outfile.write('%12d' % (i+1))
			outfile.write('\n')
			for j in range(Ny):
					outfile.write('%6d'%(j+1))
					for i in range(Nx):
							outfile.write('%12.4e'%var[i][j])
					outfile.write('\n')

with open('phi.txt', 'w') as outfile:
		print_var(outfile, phi)

with open('phi_sd.txt', 'w') as outfile:
		print_var(outfile, phi_sd)

with open('n.txt', 'w') as outfile:
		print_var(outfile, n)

with open('FOM.txt', 'w') as outfile:
		print_var(outfile, FOM)
