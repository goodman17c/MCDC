import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# =============================================================================
# Reference solution (SS)
# =============================================================================

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally/grid/x'][:]
    t = f['tally/grid/t'][:]

dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])
dt    = t[1:] - t[:-1]
t_mid = 0.5*(t[:-1]+t[1:])
K     = len(dt)
J     = len(x_mid)

data = np.load('reference.npz')
phi_t_ref = data['phi_t']
phi_ref = data['phi']

with h5py.File('output.h5', 'r') as f:
    phi      = f['tally/flux/mean'][:]
    phi_sd   = f['tally/flux/sdev'][:]
    n        = f['tally/n/mean'][:]
    n_sd     = f['tally/n/sdev'][:]
for k in range(K):
    phi[k]      /= (dx*dt[k])
    phi_sd[k]   /= (dx*dt[k])

FOM = phi*phi/phi_sd/phi_sd/np.sum(n)
FOM[n==0]=0

# =============================================================================
# Print results
# =============================================================================

def print_var(outfile, var):
    outfile.write('      ')
    outfile.write('   ix   ')
    for i in range(J):
        outfile.write('%12d' % (i+1))
    outfile.write('\n')
    outfile.write('   it ')
    outfile.write('  t/x   ')
    for i in range(J):
        outfile.write('%12.2f' % x_mid[i])
    outfile.write('\n')
    for j in range(K):
        outfile.write('%6d'%(j+1))
        outfile.write('%8.2f'%t_mid[j])
        for i in range(J):
            outfile.write('%12.4e'%var[j][i])
        outfile.write('\n')
    outfile.write('\n')

with open('output.txt', 'w') as outfile:
    outfile.write('phi\n')
    print_var(outfile, phi)
    outfile.write('phi_reference\n')
    print_var(outfile, phi_ref)
    outfile.write('phi_sd\n')
    print_var(outfile, phi_sd)
    outfile.write('n\n')
    print_var(outfile, n)
    outfile.write('FOM\n')
    print_var(outfile, FOM)

# =============================================================================
# Animate results
# =============================================================================

# Integral Flux
phi_int=np.mean(phi,1)*41
n_int=np.mean(n,1)*41
plt.plot(t_mid,phi_int,label="MC")
plt.plot(t_mid,n_int,label="n")
plt.grid()
plt.xlabel(r'$t$')
plt.ylabel(r'Integral Flux')
plt.show()


# Flux - average
fig, ax = plt.subplots()
ax.grid()
ax.set_ylim([1E-9, 1E2])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
line1, = ax.semilogy([], [],'-b',label="MC")
line2, = ax.semilogy([], [],'--r',label="Ref.")
line3, = ax.semilogy([], [],'-.g',label="FOM")
line4, = ax.semilogy([], [],':m',label="n")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()


def animate(k):
    line1.set_data(x_mid,phi[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],
                    alpha=0.2,color='b')
    line2.set_data(x_mid,phi_ref[k,:])
    line3.set_data(x_mid,FOM[k,:])
    line4.set_data(x_mid,n[k,:])
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line1, line2, line3, text


simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
simulation.save('azurv1.mp4', writer = writervideo)
plt.show()
