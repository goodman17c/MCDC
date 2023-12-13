import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import pandas as pd

method = ["A1-1", "B1-1", "C1-1"]
Np = [400, 1000, 4000, 10000]
updatelist = [0, 1, 2, 3, 10]

def process(output, updates):
    # =============================================================================
    # Reference solution (SS)
    # =============================================================================

    data = np.load("reference.npz")
    phi_ref = data["phi"]
    phi_t_ref = data["phi_t"]
    phi_x_ref = data["phi_x"]

    with h5py.File(output + ".h5", "r") as f:
        phi = f["tally/flux/mean"][:]
        phi_sd = f["tally/flux/sdev"][:]
        phit = f["tally/flux-t/mean"][:]
        phit_sd = f["tally/flux-t/sdev"][:]
        phix = f["tally/flux-x/mean"][:]
        phix_sd = f["tally/flux-x/sdev"][:]
        current = f["tally/current/mean"][:]
        current_sd = f["tally/current/sdev"][:]
        currentt = f["tally/current-t/mean"][:]
        currentt_sd = f["tally/current-t/sdev"][:]
        currentx = f["tally/current-x/mean"][:]
        currentx_sd = f["tally/current-x/sdev"][:]
        eddington = f["tally/eddington/mean"][:]
        eddington_sd = f["tally/eddington/sdev"][:]
        eddingtont = f["tally/eddington-t/mean"][:]
        eddingtont_sd = f["tally/eddington-t/sdev"][:]
        eddingtonx = f["tally/eddington-x/mean"][:]
        eddingtonx_sd = f["tally/eddington-x/sdev"][:]
        n = f["tally/n/mean"][:]
        n_t = f["tally/n-t/mean"][:]
        n_x = f["tally/n-t/mean"][:]
        ww = f["ww/center"][:]
        x = f["tally/grid/x"][:]
        t = f["tally/grid/t"][:]

    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dt = t[1:] - t[:-1]
    t_mid = 0.5 * (t[:-1] + t[1:])
    K = len(dt)
    J = len(x_mid)

    # Load deterministic iteration data
    phi_det = np.full((K, updates + 1, J), np.nan)
    phi0_det = np.full((K, J), np.nan)
    J_det = np.full((K, updates + 1, J + 1), np.nan)
    J0_det = np.full((K, J + 1), np.nan)
    Edd0_det = np.full((K, updates + 1, J), np.nan)
    Edd_det = np.full((K, updates + 1, J), np.nan)
    with open(output + "_det.txt", "r") as f:
        for k in range(K):
            for u in range(updates + 1):
                if k == 0 and u == 0:
                    continue
                # print(str(k)+" " + str(u))
                f.readline()
                # print(line)
                f.readline()
                f.readline()
                f.readline()
                for i in range(J):
                    line = f.readline().split()
                    phi0_det[k][i] = float(line[1])
                    phi_det[k][u][i] = float(line[2])
                    J0_det[k][i] = float(line[6])
                    J_det[k][u][i] = float(line[7])
                    Edd0_det[k][u][i] = float(line[3])
                    Edd_det[k][u][i] = float(line[4])
                line = f.readline().split()
                J0_det[k][J] = float(line[1])
                J_det[k][u][J] = float(line[2])
                f.readline()

    for k in range(K):
        phi[k] /= dx * dt[k]
        phi_sd[k] /= dx * dt[k]
        phit[k] /= dx
        phit_sd[k] /= dx
        phix[k] /= dt[k]
        phix_sd[k] /= dt[k]
        current[k] /= dx * dt[k]
        current_sd[k] /= dx * dt[k]
        currentt[k] /= dx
        currentt_sd[k] /= dx
        currentx[k] /= dt[k]
        currentx_sd[k] /= dt[k]
        eddington[k] /= dx * dt[k]
        eddington_sd[k] /= dx * dt[k]
        eddingtont[k] /= dx
        eddingtont_sd[k] /= dx
        eddingtonx[k] /= dt[k]
        eddingtonx_sd[k] /= dt[k]

    FOM = phi * phi / phi_sd / phi_sd / np.sum(n) / Npi
    FOM[n == 0] = 0
    newFOM = phi_ref * phi_ref / (phi - phi_ref) / (phi - phi_ref) / np.sum(n) / Npi
    newFOM[phi_ref == 0] = 0

    # Write data to large matrix
    # A column for each parameter on a single time step 4 MC scalar fluxs, ww, 3 currents, 3 eddington
    columnlist = [
        "x_mid",
        "ww",
        "phi",
        "phi_sd",
        "phi_ref",
        "phi_t",
        "phi_t_sd",
        "phi_t_ref",
        "x",
        "phi_x",
        "phi_x_sd",
        "phi_x_ref",
        "phi_det_0",
        "phi_det_1",
        "phi_det_2",
        "phi_det_3",
        "phi_det_4",
        "phi_det_5",
        "phi_det_6",
        "phi_det_7",
        "phi_det_8",
        "phi_det_9",
        "phi_det_10",
    ]
    columnlistJ = [
        "x_mid",
        "current",
        "current_sd",
        "current_t",
        "current_t_sd",
        "x",
        "current_x",
        "current_x_sd",
        "current_tx",
        "current_det_0",
        "current_det_1",
        "current_det_2",
        "current_det_3",
        "current_det_4",
        "current_det_5",
        "current_det_6",
        "current_det_7",
        "current_det_8",
        "current_det_9",
        "current_det_10",
    ]
    columnlistEdd = [
        "x_mid",
        "Eddington",
        "Eddington_sd",
        "Eddington_t",
        "Eddington_t_sd",
        "x",
        "Eddington_x",
        "Eddington_x_sd",
        "Eddington_raw_0",
        "Eddington_det_0",
        "Eddington_raw_1",
        "Eddington_det_1",
        "Eddington_raw_2",
        "Eddington_det_2",
        "Eddington_raw_3",
        "Eddington_det_3",
        "Eddington_raw_4",
        "Eddington_det_4",
        "Eddington_raw_5",
        "Eddington_det_5",
        "Eddington_raw_6",
        "Eddington_det_6",
        "Eddington_raw_7",
        "Eddington_det_7",
        "Eddington_raw_8",
        "Eddington_det_8",
        "Eddington_raw_9",
        "Eddington_det_9",
        "Eddington_raw_10",
        "Eddington_det_10",
    ]
    print(np.shape(phi[0]))
    data = np.full((K * (J + 2), 13 + updates), np.nan)
    dataJ = np.full((K * (J + 2), 10 + updates), np.nan)
    dataEdd = np.full((K * (J + 2), 10 + 2 * updates), np.nan)
    for k in range(K):
        data[k * (J + 2), 0] = t_mid[k]
        for i in range(J):
            data[k * (J + 2) + i + 1, 0] = x_mid[i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 1] = ww[k][i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 2] = phi[k][i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 3] = phi_sd[k][i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 4] = phi_ref[k][i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 5] = phit[k + 1][i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 6] = phit_sd[k + 1][i]
        for i in range(J):
            data[k * (J + 2) + i + 1, 7] = phi_t_ref[k][i]
        for i in range(J + 1):
            data[k * (J + 2) + i + 1, 8] = x[i]
        for i in range(J + 1):
            data[k * (J + 2) + i + 1, 9] = phix[k][i]
        for i in range(J + 1):
            data[k * (J + 2) + i + 1, 10] = phix_sd[k][i]
        for i in range(J + 1):
            data[k * (J + 2) + i + 1, 11] = phi_x_ref[k][i]
        for u in range(updates + 1):
            for i in range(J):
                data[k * (J + 2) + i + 1, 12 + u] = phi_det[k][u][i]

        dataJ[k * (J + 2), 0] = t_mid[k]
        for i in range(J):
            dataJ[k * (J + 2) + i + 1, 0] = x_mid[i]
        for i in range(J):
            dataJ[k * (J + 2) + i + 1, 1] = current[k][i][0]
        for i in range(J):
            dataJ[k * (J + 2) + i + 1, 2] = current_sd[k][i][0]
        for i in range(J):
            dataJ[k * (J + 2) + i + 1, 3] = currentt[k + 1][i][0]
        for i in range(J):
            dataJ[k * (J + 2) + i + 1, 4] = currentt_sd[k + 1][i][0]
        for i in range(J + 1):
            dataJ[k * (J + 2) + i + 1, 5] = x[i]
        for i in range(J + 1):
            dataJ[k * (J + 2) + i + 1, 6] = currentx[k][i][0]
        for i in range(J + 1):
            dataJ[k * (J + 2) + i + 1, 7] = currentx_sd[k][i][0]
        for i in range(J + 1):
            dataJ[k * (J + 2) + i + 1, 8] = J0_det[k][i]
        for u in range(updates + 1):
            for i in range(J + 1):
                dataJ[k * (J + 2) + i + 1, 9 + u] = J_det[k][u][i]

        dataEdd[k * (J + 2), 0] = t_mid[k]
        for i in range(J):
            dataEdd[k * (J + 2) + i + 1, 0] = x_mid[i]
        for i in range(J):
            dataEdd[k * (J + 2) + i + 1, 1] = eddington[k][i][0] / phi[k][i]
        for i in range(J):
            dataEdd[k * (J + 2) + i + 1, 2] = eddington_sd[k][i][0] / phi[k][i]
        for i in range(J):
            dataEdd[k * (J + 2) + i + 1, 3] = eddingtont[k + 1][i][0] / phit[k + 1][i]
        for i in range(J):
            dataEdd[k * (J + 2) + i + 1, 4] = (
                eddingtont_sd[k + 1][i][0] / phit[k + 1][i]
            )
        for i in range(J + 1):
            dataEdd[k * (J + 2) + i + 1, 5] = x[i]
        for i in range(J + 1):
            dataEdd[k * (J + 2) + i + 1, 6] = eddingtonx[k][i][0] / phix[k][i]
        for i in range(J + 1):
            dataEdd[k * (J + 2) + i + 1, 7] = eddingtonx_sd[k][i][0] / phix[k][i]
        for u in range(updates + 1):
            for i in range(J):
                dataEdd[k * (J + 2) + i + 1, 8 + 2 * u] = Edd0_det[k][u][i]
                dataEdd[k * (J + 2) + i + 1, 8 + 2 * u + 1] = Edd_det[k][u][i]

    # =============================================================================
    # Print results
    # =============================================================================

    with pd.ExcelWriter(output + ".xlsx") as writer:
        # Complete data in time and space
        df = pd.DataFrame(data, columns=columnlist[: (13 + updates)])
        df.to_excel(writer, sheet_name="Scalar Flux")
        df = pd.DataFrame(dataJ, columns=columnlistJ[: (10 + updates)])
        df.to_excel(writer, sheet_name="Current")
        df = pd.DataFrame(dataEdd, columns=columnlistEdd[: (10 + 2 * updates)])
        df.to_excel(writer, sheet_name="Eddington")
        df = pd.DataFrame(phi)
        df.to_excel(writer, sheet_name="Plotting")

        # Data normed over space

        # Copy deterministic statistic from file for each update (predictor or corrector)

        # Run Overview statistics and flags


for Npi in Np:
    for fj in range(1, 2):
        for updates in updatelist:
            for methodi in method:
                output = (
                    methodi + "_" + str(2**fj) + "_" + str(updates) + "_" + str(Npi)
                )
                process(output, updates)
