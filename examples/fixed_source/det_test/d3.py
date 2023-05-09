import deterministic as det
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=2e7)

##Problem Specifications
det.t = np.linspace(0, 0.4, 21)  # 20 time steps to 0.4 ns
det.x = np.linspace(0, 5, 26)  # 25 spatial steps to 5 cm
det.v = 2.9979e1  # cm/ns
det.sigmaT = 1  # cm-1
det.sigmaS = 0  # cm-1
det.q = 0
det.IC = 1e-3  # angular flux in n/ns/cm2/angle cosine
det.leftBC = 1
det.rightBC = 0
det.M = 8
det.eps_phi = 1e-8
det.delta = 0  # infintesimal to prevent divide by zeros

det.method = 1

det.QD()
det.write("d3/d3.txt")
det.tswrite("d3/")
