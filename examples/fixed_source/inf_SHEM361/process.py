import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('SHEM-361.npz') as data:
    data     = np.load('SHEM-361.npz')
    SigmaT   = data['SigmaT'] + data['SigmaC']
    nuSigmaF = data['nuSigmaF']
    SigmaS   = data['SigmaS']
    E        = data['E']

G = len(SigmaT)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

A = np.diag(SigmaT) - SigmaS - nuSigmaF
Q = np.ones(G)/G

phi_exact = np.linalg.solve(A,Q)

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    phi = f['tally/flux/mean'][:]
    phi_sd = f['tally/flux/sdev'][:]
   
phi       *= E_mid/dE
phi_sd    *= E_mid/dE
phi_exact *= E_mid/dE
plt.step(E_mid,phi,'-b',where='mid',label="MC")
plt.fill_between(E_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b',step='mid')
plt.step(E_mid,phi_exact,'--r',where='mid',label='analytical')
plt.xscale('log')
plt.xlabel(r'$E$, eV')
plt.ylabel(r'$E\phi(E)$')
plt.grid()
plt.legend()
plt.show()