import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

ref=["reference.npz", "reference_40.npz", "reference_80.npz"]

def add_to_plot(i, j, k):
		method = ["1a","1b","1c","1d"]
		width = ["_1", "_2", "_4", "_8"]
		Nt = ["_20","_40","_80"]
		# =============================================================================
		# Reference solution (SS)
		# =============================================================================

		# Load grids
		data = np.load(ref[k])
		phi_t_ref = data['phi_t']
		phi_ref = data['phi']
		x=data['x']
		t=data['t']

		dx    = (x[1]-x[0])
		x_mid = 0.5*(x[:-1]+x[1:])
		dt    = t[1:] - t[:-1]
		t_mid = 0.5*(t[:-1]+t[1:])
		K     = len(dt)
		J     = len(x_mid)
		file = method[i] + width[j] + Nt[k]+".h5"
		with h5py.File(file, 'r') as f:
				phi      = f['tally/flux/mean'][:]
				phi_sd   = f['tally/flux/sdev'][:]
				n        = f['tally/n/mean'][:]
				n_t     = f['tally/n-t/mean'][:]
				for k in range(K):
						phi[k]      /= (dx*dt[k])
						phi_sd[k]   /= (dx*dt[k])

				FOM = phi_ref*phi_ref/phi_sd/phi_sd/np.sum(n)
				FOM[n==0]=0
				n_int=np.sum(n,1)
				n_t_int=np.sum(n_t,1)

				max_rel_err=np.zeros(K)
				err_inf =np.zeros(K)
				err_L1=np.zeros(K)
				err_L2=np.zeros(K)
				err_2=np.zeros(K)
				rel_err_inf=np.zeros(K)
				rel_err_L1=np.zeros(K)
				rel_err_L2=np.zeros(K)
				rel_err_2=np.zeros(K)
				stat_err=np.zeros(K)
				
				FOM_L2=np.zeros(K)

				#Analysis of Numerical Solutions
				for k in range(K):
						max_rel_err[k] = np.max(np.abs(1-np.nan_to_num(phi[k]/phi_ref[k],nan=1) ))
						err_inf[k] = np.max(np.abs(phi[k]-phi_ref[k]))
						err_L1[k] = np.sum(np.abs(phi[k]-phi_ref[k])*dx)
						err_L2[k] = np.sqrt(np.sum(np.power(phi[k]-phi_ref[k],2)*dx))
						err_2[k] = np.sqrt(np.sum(np.power(phi[k]-phi_ref[k],2)))
						rel_err_inf[k] = err_inf[k]/np.max(np.abs(phi_ref[k]))
						rel_err_L1[k] = err_L1[k]/np.sum(np.abs(phi_ref[k])*dx)
						rel_err_L2[k] = err_L2[k]/np.sqrt(np.sum(np.power(phi_ref[k],2)*dx))
						rel_err_2[k] = err_2[k]/np.sqrt(np.sum(np.power(phi_ref[k],2)))
						stat_err[k] = np.sqrt(np.sum(np.power(phi_sd[k],2))/np.sum(np.power(phi[k],2)))
						
						FOM_L2[k] = np.sqrt(np.sum(np.power(FOM[k],2)*dx))

				plt.semilogy(t_mid,err_L2,label=file)
				#plt.semilogy(t_mid,err_L2,label=file)

# =============================================================================
# Animate results
# =============================================================================

for i in range(4):
	for j in range(2):
		add_to_plot(3,i,j)
plt.grid()
plt.legend()
plt.xlabel(r'$t$')
plt.show()
