import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import matplotlib.animation as animation


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("outputIC.h5", "r") as f:
    x = f["tally/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = f["tally/grid/y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    t = f["tally/grid/t"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(y, x)

    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    n = f["tally/n/mean"][:]
    rt = f["runtime"][:]

with h5py.File("outputWW.h5", "r") as f:
    x = f["tally/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = f["tally/grid/y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    t = f["tally/grid/t"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(y, x)

    phi2 = f["tally/flux/mean"][:]
    phi_sd2 = f["tally/flux/sdev"][:]
    n2 = f["tally/n/mean"][:]
    rt2 = f["runtime"][:]

FOM = phi*phi/phi_sd/phi_sd/rt
FOM2 = phi2*phi2/phi_sd2/phi_sd2/rt2

error = np.sqrt(np.nansum(phi_sd*phi_sd/phi/phi,(1,2)))
error2 = np.sqrt(np.nansum(phi_sd2*phi_sd2/phi2/phi2,(1,2)))

L2FOM = 1/rt/error
L2FOM2 = 1/rt2/error2

print(error)
print(np.nanmean(FOM))
plt.semilogy(t_mid,L2FOM)
plt.semilogy(t_mid,L2FOM2)
plt.xlabel('t')
plt.ylabel('FOM')
plt.legend(["Implicit Capture", "Weight Window k=1"])
plt.show()
# n limits
#cmin = 6E-3
#cmax = 2E-2
# FOM limits
cmin = .5
cmax = 500

fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, FOM[9], norm=LogNorm(vmin=cmin, vmax=cmax))
#cax = ax.pcolormesh(X, Y, np.log(n[0]), vmin=np.nanmin(np.log(n[n != 0])), vmax=np.nanmax(np.log(n)))
text = ax.text(0.02, 1.02, "", transform=ax.transAxes)
ax.set_aspect("equal", "box")
ax.set_xlabel("$y$ [cm]")
ax.set_ylabel("$x$ [cm]")
fig.colorbar(cax)
text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[9], t[9 + 1]))
plt.show()

fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, FOM2[9], norm=LogNorm(vmin=cmin, vmax=cmax))
#cax = ax.pcolormesh(X, Y, np.log(n[0]), vmin=np.nanmin(np.log(n[n != 0])), vmax=np.nanmax(np.log(n)))
text = ax.text(0.02, 1.02, "", transform=ax.transAxes)
ax.set_aspect("equal", "box")
ax.set_xlabel("$y$ [cm]")
ax.set_ylabel("$x$ [cm]")
fig.colorbar(cax)
text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[9], t[9 + 1]))
plt.show()

#def animate(i):
#    cax.set_array(FOM[i])
    #cax.set_clim(np.nanmin(np.log(n[i][n[i] != 0])), np.nanmax(np.log(n[i])))
#    text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[i], t[i + 1]))


#anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
#plt.show()
