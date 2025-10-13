import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loaddata(fname, NoPlot=True, errname=None):
    data = pd.read_excel(fname)
    names = []
    pops = []

    for col in data.columns:
        if col.lower() == 'time':
            tlist = np.array(data[col])
        else:
            names.append(col.upper())
            pops.append(data[col])

    pops = np.array(pops)

    if errname:
        errdata = pd.read_excel(errname)
        errs = np.array([errdata[col] for col in errdata.columns])
    else:
        errs = np.sqrt(pops) + pops * 0.1 + 1

    if not NoPlot:
        plt.figure()
        for i in range(len(names)):
            plt.plot(tlist, pops[i], label=names[i])
        plt.legend()

    return tlist, names, pops, errs
