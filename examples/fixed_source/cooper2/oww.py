import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================
v = 1.383
pi = math.acos(-1)
# Results
with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = f["tally/grid/y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    Nx = len(x_mid)
    Ny = len(y_mid)

    cf = 5 * 5

    phi = f["tally/flux/mean"][:] * 4 * cf
    phi_octant = f["tally/octant-flux/mean"][:] * 4 * cf

phi2 = np.repeat(np.expand_dims(phi, 2), 8, 2)
kfact = -1.0

ww = phi2 * np.power(8 * phi_octant / phi2, kfact)
ww[phi2 == 0] = 0
ww /= np.max(phi)  # used a fixed normalization constant

np.savez("ww.npz", phi=ww)


def print_var(outfile, var):
    outfile.write(" iy/ix")
    for i in range(Nx):
        outfile.write("%12d" % (i + 1))
    outfile.write("\n")
    for j in range(Ny):
        outfile.write("%6d" % (j + 1))
        for i in range(Nx):
            outfile.write("%12.4e" % var[i][j])
        outfile.write("\n")


# write file for octant 1
for o in range(8):
    with open("ww%d.txt" % (o + 1), "w") as outfile:
        print_var(outfile, ww[:, :, o])

with open("ww.txt", "w") as outfile:
    print_var(outfile, phi)
