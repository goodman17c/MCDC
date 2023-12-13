import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

Nt = [20, 40, 80]
ref = ["reference.npz", "reference_40.npz", "reference_80.npz"]


method = ["A1-1", "B1-1", "C1-1", "F1-1"]#, "A1-1int", "A1-1lin"]
methodname = [
    "Method A1-1",
    "Method B1-1",
    "Method C1-1",
    "McClaren",
#    "Method A1-1 Interpolated Eddington",
#    "Method A1-1 Linear Interp",
]
Np = [400, 1000, 4000, 10000]
updatelist = [0, 1, 2, 3, 10]

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
    L2_rel_err_list = []
    

    def add_out_to_stat_lists(output, Npart):
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
            phi[k] /= (dx*dt[k])
            #phi[k] /= dt[k]
            phi_sd[k] /= (dx*dt[k])
            #phi_sd[k] /= dt[k]

        error = phi-phi_ref
        phi2_ref = np.maximum(phi_ref, phi)
        rel_error = error/phi2_ref
        FOM = 1 / rel_error / rel_error / np.sum(n) / Npart
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
        L2_rel_err = np.zeros(K)

        # Analysis of Numerical Solutions
        for k in range(K):
            L2_rel_err[k] = np.sqrt(np.nansum(rel_error[k]*rel_error[k]*dx))
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
            FOM_L2[k] = 1 / L2_rel_err[k] / np.sum(n) / Npart

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
        L2_rel_err_list.append(L2_rel_err)

    for Npi in Np:
        try:
            add_out_to_stat_lists("Analog_" + str(Npi), Npi)
            index_list.append("Analog" + " particles=" + str(Npi))
        except:
            pass

    for Npi in Np:
        try:
            add_out_to_stat_lists("IC_" + str(Npi), Npi)
            index_list.append("Implicit Capture" + " particles=" + str(Npi))
        except:
            pass
    
    for Npi in Np:
        try:
            add_out_to_stat_lists("BC_" + str(Npi), Npi)
            index_list.append("Branchless Collision" + " particles=" + str(Npi))
        except:
            pass

    for methodi in range(len(method)):
        for fj in range(3):
            for Npi in Np:
                for updates in updatelist:
                    output = (
                        method[methodi]
                        + "_"
                        + str(2**fj)
                        + "_"
                        + str(updates)
                        + "_"
                        + str(Npi)
                    )
                    try:
                        add_out_to_stat_lists(output, Npi)
                        index_list.append(
                            methodname[methodi]
                            + " updates="
                            + str(updates)
                            + " width="
                            + str(2**fj)
                            + " particles="
                            + str(Npi)
                        )
                    except:
                        pass

    with pd.ExcelWriter("Summary_" + str(Nt[Nti]) + ".xlsx") as writer:
        df = pd.DataFrame(FOM_L2_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="Figure of Merit L2 Norm")

        df = pd.DataFrame(
            L2_rel_err_list, index=index_list, columns=range(1, Nt[Nti] + 1)
        )
        df.to_excel(writer, sheet_name="L2 norm of relative error")
        
        df = pd.DataFrame(n_t_int_list, index=index_list, columns=range(1, Nt[Nti] + 1))
        df.to_excel(writer, sheet_name="n_t")

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

