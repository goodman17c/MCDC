import numpy as np
import matplotlib.pyplot as plt
import deterministic as det

np.set_printoptions(threshold=2e7)

##Problem Specifications
det.t = np.linspace(0, 1, 51)  # 50 time steps to 1 ns
det.x = np.linspace(0, 5, 101)  # 100 spatial steps to 5 cm
det.v = 2.9979e1  # cm/ns
det.sigmaT = 0.1  # cm-1
det.sigmaS = 0.05  # cm-1
det.q = 0
det.IC = 1e-3  # angular flux in n/ns/cm2/angle cosine
det.leftBC = 1e2
det.rightBC = 0
# angular quadrature
det.M = 8

det.delta = 0  # infintesimal to prevent divide by zeros

det.eps_phi = 1e-10

det.method = 1  # 0 is SI transport, 1 is QD

det.QD()
det.write("d1.txt")
det.tswrite("")

x = np.squeeze(det.x)
x2 = (x[1:] + x[:-1]) / 2

plt.semilogy(x2, np.transpose(det.phi))
plt.show()
