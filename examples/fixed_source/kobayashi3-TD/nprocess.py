import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import matplotlib.animation as animation


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = f["tally/grid/y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    t = f["tally/grid/t"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(y, x)

    phi = f["tally/flux/mean"][:]
    n = f["tally/n/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

vmin = np.nanmin(np.log10(n[n > 0]))
vmax = np.nanmax(np.log10(n))

fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, np.log10(n[0]), vmin=vmin, vmax=vmax)
text = ax.text(0.02, 1.02, "", transform=ax.transAxes)
ax.set_aspect("equal", "box")
ax.set_xlabel("$y$ [cm]")
ax.set_ylabel("$x$ [cm]")
cbar = plt.colorbar(cax)


def animate(i):
    #    cbar.remove()
    cax.set_array(np.log10(n[i]))
    cax.set_clim(vmin, vmax)
    #    cbar = plt.colorbar(cax)
    text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[i], t[i + 1]))


anim = animation.FuncAnimation(fig, animate, interval=50, frames=len(t) - 1)
anim.save("kobay_n.gif", savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0})
plt.show()
