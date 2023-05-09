import numpy as np
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

data = np.load("reference.npz")
phi_ref = data["phi"]
phi_t_ref = data["phi_t"]
phi_t_ref[1:] = phi_t_ref[:-1]
phi_t_ref[0, :] = 0
phi_t_ref[0, 100] = 1

for k in range(K):
    phi_ref[k] = phi_ref[k] / np.max(phi_ref[k])
    phi_t_ref[k] = phi_t_ref[k] / np.max(phi_t_ref[k])

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


with open("WWs.txt", "w") as outfile:
    outfile.write("Good WW\n")
    print_var(outfile, phi_ref)
    phi_ref[1:] = phi_ref[:-1]  # Use weight windows from previous time step
    outfile.write("WW behind 1 time step\n")
    print_var(outfile, phi_ref)
    outfile.write("WW from solution at beginning of time step\n")
    print_var(outfile, phi_t_ref)
