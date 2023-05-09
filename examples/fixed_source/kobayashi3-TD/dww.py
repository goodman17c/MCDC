import numpy as np
import h5py

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
    J = f["tally/current/mean"][:]


Nx = len(x_mid)
Ny = len(y_mid)
K = len(t_mid)

Jx = J[:, :, :, 0]
Jy = J[:, :, :, 1]
Jz = J[:, :, :, 2]

kfact = -1.0

Bx = np.zeros_like(phi)
By = np.zeros_like(phi)
Bz = np.zeros_like(phi)
B = np.zeros_like(phi)

lcap = 0.6

for k in range(K):
    lx = Jx[k] / phi[k]
    ly = Jy[k] / phi[k]
    lz = Jz[k] / phi[k]
    lx[np.isnan(lx)] = 0
    ly[np.isnan(ly)] = 0
    lz[np.isnan(lz)] = 0
    l = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx[l > lcap] = lx[l > lcap] / l[l > lcap] * lcap
    ly[l > lcap] = ly[l > lcap] / l[l > lcap] * lcap
    lz[l > lcap] = lz[l > lcap] / l[l > lcap] * lcap
    l = np.sqrt(lx * lx + ly * ly + lz * lz)
    # print(np.max(l))
    Bt = 3 * l  # initial guess
    # Euler's Method inversion
    for t in range(5):
        Bt[Bt > 0] = Bt[Bt > 0] - (
            np.cosh(Bt[Bt > 0]) / np.sinh(Bt[Bt > 0]) - 1 / Bt[Bt > 0] - l[Bt > 0]
        ) / (
            1 / Bt[Bt > 0] / Bt[Bt > 0] - 1 / np.sinh(Bt[Bt > 0]) / np.sinh(Bt[Bt > 0])
        )
    # err=l-np.cosh(Bt)/np.sinh(Bt)+1/Bt
    # err[l==0]=0
    l[l == 0] = 1
    B[k] = Bt
    Bx[k] = lx / l * B[k]
    By[k] = ly / l * B[k]
    Bz[k] = lz / l * B[k]

ww = phi
ww[B > 0] *= np.power(B[B > 0] / np.sinh(B[B > 0]), kfact)
ww /= np.max(ww[0])  # used a fixed normalization constant

Bx *= kfact
By *= kfact
Bz *= kfact
B *= abs(kfact)

# with np.printoptions(threshold=np.inf):
# 	print(B[0])
np.savez("ww.npz", ww=ww, Bx=Bx, By=By, Bz=Bz)


def print_var(outfile, var):
    for k in range(K):
        outfile.write("Time Step " + str(k + 1) + "\n")
        outfile.write(" iy/ix")
        for i in range(Nx):
            outfile.write("%12d" % (i + 1))
        outfile.write("\n")
        for j in range(Ny):
            outfile.write("%6d" % (j + 1))
            for i in range(Nx):
                outfile.write("%12.4e" % var[k][i][j])
            outfile.write("\n")


# write file for ww
with open("ww.txt", "w") as outfile:
    print_var(outfile, ww)

# write file for Bx
with open("Bx.txt", "w") as outfile:
    print_var(outfile, Bx)

# write file for By
with open("By.txt", "w") as outfile:
    print_var(outfile, By)

# write file for Bz
with open("Bz.txt", "w") as outfile:
    print_var(outfile, Bz)

# write file for B
with open("B.txt", "w") as outfile:
    print_var(outfile, B)
