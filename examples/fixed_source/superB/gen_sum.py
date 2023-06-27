import numpy as np
import h5py
import pandas as pd

Nt = [20, 40, 80]
ref = ["reference.npz", "reference_40.npz", "reference_80.npz"]


method = ["8", "11", "18", "21"]
methodname = [
    "Semi-Implicit LOQD full",
    "Semi-Implicit LOQD half",
    "Semi-Implicit LOQD full corrector",
    "Semi-Implicit LOQD half corrector",
]
Np = [400, 1000, 4000, 10000, 40000]

for Nti in range(1):
    data = np.load(ref[Nti])
    phi_ref = data["phi"]

    index_list = []

    max_rel_err_list = []
    err_inf_list = []
    err_L1_list = []
    err_L2_list = []
    err_2_list = []
    rel_err_inf_list = []
    rel_err_L1_list = []
    rel_err_L2_list = []
    rel_err_2_list = []
    stat_err_list = []
    n_t_int_list = []
    FOM_L2_list = []
    newFOM_L1_list = []
    newFOM_Linf_list = []

    def add_out_to_stat_lists(output):
        with h5py.File(output + ".h5", "r") as f:
            phi = f["tally/flux/mean"][:]
            phi_sd = f["tally/flux/sdev"][:]
            n = f["tally/n/mean"][:]
            n_t = f["tally/n-t/mean"][1:]
            x = f["tally/grid/x"][:]
            t = f["tally/grid/t"][:]

        dx = x[1] - x[0]
        x_mid = 0.5 * (x[:-1] + x[1:])
        dt = t[1:] - t[:-1]
        t_mid = 0.5 * (t[:-1] + t[1:])
        K = len(dt)
        J = len(x_mid)
        for k in range(K):
            phi[k] /= dx * dt[k]
            phi_sd[k] /= dx * dt[k]

        FOM = phi * phi / phi_sd / phi_sd / np.sum(n) / Npi
        FOM[n == 0] = 0
        newFOM = phi_ref * phi_ref / (phi - phi_ref) / (phi - phi_ref) / np.sum(n) / Npi
        newFOM[phi_ref == 0] = 0

        # Calculate Integral quatities
        phi_int = np.mean(phi, 1) * 40.2
        n_int = np.sum(n, 1)
        n_t_int = np.sum(n_t, 1)

        max_rel_err = np.zeros(K)
        err_inf = np.zeros(K)
        err_L1 = np.zeros(K)
        err_L2 = np.zeros(K)
        err_2 = np.zeros(K)
        rel_err_inf = np.zeros(K)
        rel_err_L1 = np.zeros(K)
        rel_err_L2 = np.zeros(K)
        rel_err_2 = np.zeros(K)
        stat_err = np.zeros(K)
        FOM_L2 = np.zeros(K)
        newFOM_L1 = np.zeros(K)
        newFOM_Linf = np.zeros(K)

        # Analysis of Numerical Solutions
        for k in range(K):
            max_rel_err[k] = np.max(
                np.abs(1 - np.nan_to_num(phi[k] / phi_ref[k], nan=1))
            )
            err_inf[k] = np.max(np.abs(phi[k] - phi_ref[k]))
            err_L1[k] = np.sum(np.abs(phi[k] - phi_ref[k]) * dx)
            err_L2[k] = np.sqrt(np.sum(np.power(phi[k] - phi_ref[k], 2) * dx))
            err_2[k] = np.sqrt(np.sum(np.power(phi[k] - phi_ref[k], 2)))
            rel_err_inf[k] = err_inf[k] / np.max(np.abs(phi_ref[k]))
            rel_err_L1[k] = err_L1[k] / np.sum(np.abs(phi_ref[k]) * dx)
            rel_err_L2[k] = err_L2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2) * dx))
            rel_err_2[k] = err_2[k] / np.sqrt(np.sum(np.power(phi_ref[k], 2)))
            stat_err[k] = np.sqrt(
                np.sum(np.power(phi_sd[k], 2)) / np.sum(np.power(phi[k], 2))
            )
            FOM_L2[k] = np.sqrt(np.sum(np.power(FOM[k], 2) * dx))
            newFOM_L1[k] = 1 / np.sum(1 / newFOM[k][newFOM[k] > 0] * dx)
            newFOM_Linf[k] = 1 / np.max(1 / newFOM[k][newFOM[k] > 0] * dx)

        max_rel_err_list.append(max_rel_err)
        err_inf_list.append(err_inf)
        err_L1_list.append(err_L1)
        err_L2_list.append(err_L2)
        err_2_list.append(err_2)
        rel_err_inf_list.append(rel_err_inf)
        rel_err_L1_list.append(rel_err_L1)
        rel_err_L2_list.append(rel_err_L2)
        rel_err_2_list.append(rel_err_2)
        stat_err_list.append(stat_err)
        n_t_int_list.append(n_t_int)
        FOM_L2_list.append(FOM_L2)
        newFOM_L1_list.append(newFOM_L1)
        newFOM_Linf_list.append(newFOM_Linf)

    for Npi in Np:
        try:
            add_out_to_stat_lists("1_" + str(Nt[Nti]) + "_" + str(Npi))
            index_list.append("Implicit Capture" + " particles=" + str(Npi))
        except:
            pass

    for methodi in range(len(method)):
        for fj in range(3):
            for Npi in Np:
                output = (
                    method[methodi]
                    + "_"
                    + str(2**fj)
                    + "_"
                    + str(Nt[Nti])
                    + "_"
                    + str(Npi)
                )
                try:
                    add_out_to_stat_lists(output)
                    index_list.append(
                        methodname[methodi]
                        + " width="
                        + str(2**fj)
                        + " particles="
                        + str(Npi)
                    )
                except:
                    pass

    for Npi in Np:
        try:
            add_out_to_stat_lists("0_" + str(Nt[Nti]) + "_" + str(Npi))
            index_list.append("Analog" + " particles=" + str(Npi))
        except:
            pass

    for fj in range(3):
        for Npi in Np:
            try:
                output = "10" + "_" + str(2**fj) + "_" + str(Nt[Nti]) + "_" + str(Npi)
                add_out_to_stat_lists(output)
                index_list.append(
                    "Middle of TS" + " width=" + str(2**fj) + " particles=" + str(Npi)
                )
            except:
                pass

    with pd.ExcelWriter("Summary_" + str(Nt[Nti]) + ".xlsx") as writer:
        df = pd.DataFrame(n_t_int_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="n_t")

        df = pd.DataFrame(FOM_L2_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="Figure of Merit (ideal) L2 Norm")

        df = pd.DataFrame(
            stat_err_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Statistical Error")

        df = pd.DataFrame(
            max_rel_err_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Maximum Relative Error")

        df = pd.DataFrame(err_inf_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="Error Infinity Norm")

        df = pd.DataFrame(err_L1_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="Error L1 Norm")

        df = pd.DataFrame(err_L2_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="Error L2 Norm")

        df = pd.DataFrame(err_2_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="Error 2 Norm")

        df = pd.DataFrame(
            rel_err_inf_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Relative Error Infinity Norm")

        df = pd.DataFrame(
            rel_err_L1_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Relative Error L1 Norm")

        df = pd.DataFrame(
            rel_err_L2_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Relative Error L2 Norm")

        df = pd.DataFrame(
            rel_err_2_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Relative Error 2 Norm")

        df = pd.DataFrame(
            newFOM_L1_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Figure of Merit (error) L1")

        df = pd.DataFrame(
            newFOM_Linf_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="Figure of Merit (error) Linf")
