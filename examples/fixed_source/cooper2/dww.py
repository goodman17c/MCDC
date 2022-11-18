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
	J = f['tally/current/mean'][:]*4*cf

Jx = J[:,:,0]
Jy = J[:,:,1]

kfact = -1.0

Bx=np.zeros_like(phi)
By=np.zeros_like(phi)
Bz=np.zeros_like(phi)
B=np.zeros_like(phi)

lx = Jx/phi
ly = Jy/phi
l = np.sqrt(lx*lx+ly*ly)
B = 3*l #initial guess
#Euler's Method inversion
for t in range(5):
	B=B-(np.cosh(B)/np.sinh(B)-1/B-l)/(1/B/B-1/np.sinh(B)/np.sinh(B))
err=l-np.cosh(B)/np.sinh(B)+1/B
#err[l==0]=0
#l[l==0]=1
Bx = lx/l*B
By = ly/l*B

ww = phi
#ww[B>0] *= B[B>0]/np.sinh(B[B>0])
ww *= np.power(B/np.sinh(B),kfact)
ww /= np.max(ww) # used a fixed normalization constant

Bx *= kfact
By *= kfact
B *= abs(kfact)

#with np.printoptions(threshold=np.inf):
	#print(B)

print(np.max(B))

np.savez("ww.npz",phi=ww,Bx=Bx,By=By,Bz=Bz)

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

#write file for ww
with open('ww.txt', 'w') as outfile:
    print_var(outfile, ww)

#write file for Bx
with open('Bx.txt', 'w') as outfile:
    print_var(outfile, Bx)

#write file for By
with open('By.txt', 'w') as outfile:
    print_var(outfile, By)

#write file for B
with open('B.txt', 'w') as outfile:
    print_var(outfile, B)
