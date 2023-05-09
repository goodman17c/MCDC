import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# =============================================================================
# Reference solution (SS)
# =============================================================================

# Load grids
with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    t = f["tally/grid/t"][:]

dx = x[1] - x[0]
x_mid = 0.5 * (x[:-1] + x[1:])
dt = t[1:] - t[:-1]
t_mid = 0.5 * (t[:-1] + t[1:])
K = len(dt)
J = len(x_mid)

# data = np.load('reference.npz')
# phi_t_ref = data['phi_t']
# phi_ref = data['phi']

ss = 1e-3 * 5 * 2 + 1e2 * 1 * 2

with h5py.File("output.h5", "r") as f:
    phi = f["tally/flux/mean"][:] * ss
    phi_sd = f["tally/flux/sdev"][:] * ss
    n = f["tally/n/mean"][:]
    n_t = f["tally/n-t/mean"][:]
for k in range(K):
    phi[k] /= dx * dt[k]
    phi_sd[k] /= dx * dt[k]

FOM = phi * phi / phi_sd / phi_sd / np.sum(n)
FOM[n == 0] = 0

# Average weight of particles in cell
w_avg = phi / n
w_avg[n == 0] = 0

# Calculate Integral quatities
phi_int = np.mean(phi, 1) * 40.2
n_int = np.sum(n, 1)
n_t_int = np.sum(n_t, 1)

# =============================================================================
# Print results
# =============================================================================


def print_var(outfile, var):
    outfile.write("      ")
    outfile.write("   ix   ")
    for i in range(J):
        outfile.write("%12d" % (i + 1))
    outfile.write("\n")
    outfile.write("   it ")
    outfile.write("  t/x   ")
    for i in range(J):
        outfile.write("%12.2f" % x_mid[i])
    outfile.write("\n")
    for j in range(K):
        outfile.write("%6d" % (j + 1))
        outfile.write("%8.2f" % t_mid[j])
        for i in range(J):
            outfile.write("%12.4e" % var[j][i])
        outfile.write("\n")
    outfile.write("\n")


def print_int(outfile, var):
    for i in range(K):
        outfile.write("%12.4e" % var[i])
    outfile.write("\n")


with open("output.txt", "w") as outfile:
    # Integral Quantities
    outfile.write("\n       it   ")
    for i in range(K):
        outfile.write("%12d" % (i + 1))
    outfile.write("\n")
    outfile.write("       t    ")
    print_int(outfile, t_mid)
    outfile.write("integral phi")
    print_int(outfile, phi_int)
    outfile.write("integral n  ")
    print_int(outfile, n_int)
    outfile.write("integral n_t")
    print_int(outfile, n_int)

    # Space and Time data
    outfile.write("\n")
    outfile.write("phi\n")
    print_var(outfile, phi)
    #    outfile.write('phi_reference\n')
    #    print_var(outfile, phi_ref)
    outfile.write("phi_sd\n")
    print_var(outfile, phi_sd)
    outfile.write("n\n")
    print_var(outfile, n)
    outfile.write("FOM\n")
    print_var(outfile, FOM)
    outfile.write("n_t\n")
    print_var(outfile, n_t[1:, :])

# =============================================================================
# Animate results
# =============================================================================

# Integral Flux
# plt.plot(t_mid,phi_int,label="MC")
# plt.plot(t_mid,n_int,label="n")
# plt.grid()
# plt.xlabel(r'$t$')
# plt.ylabel(r'Integral Flux')
# plt.show()


# Flux - average
fig, ax = plt.subplots()
ax.grid()
ax.set_ylim([1e-12, 1e2])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Flux")
(line1,) = ax.semilogy([], [], "-b", label="MC")
(line2,) = ax.semilogy([], [], "--r", label="Ref.")
(line3,) = ax.semilogy([], [], "-.g", label="FOM")
(line4,) = ax.semilogy([], [], ":m", label="n")
text = ax.text(0.02, 0.9, "", transform=ax.transAxes)
ax.legend()


def animate(k):
    line1.set_data(x_mid, phi[k, :])
    ax.collections.clear()
    ax.fill_between(
        x_mid, phi[k, :] - phi_sd[k, :], phi[k, :] + phi_sd[k, :], alpha=0.2, color="b"
    )
    line2.set_data(x_mid, phi_ref[k, :])
    line3.set_data(x_mid, FOM[k, :])
    line4.set_data(x_mid, n[k, :])
    text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[k], t[k + 1]))
    return line1, line2, line3, text


simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
simulation.save("azurv1.mp4", writer=writervideo)
plt.show()
