import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def make_init(names, pops, specs, dct):
    y0 = np.zeros(len(specs))
    for i, name in enumerate(names):
        if name in dct:
            y0[dct[name]] = pops[i][0]
    return y0

def vec(t,y,par,rs,specs,dct):                               #Calculates vector of rates for all species
    res=np.zeros(len(specs))
    n=0
    for i in rs:
        rate=i.rate(t,y,par,dct)
        for j in i.products:
            res[dct[j]]=res[dct[j]]+rate
        for j in i.reactants:
            res[dct[j]]=res[dct[j]]-rate
        n+=1
    return res

def jac(t,y,par,rs,specs,dct):                               #Calculates Jacobian matrix of rates with respect to species concentrations
    res=np.zeros((len(specs),len(specs)))
    n=0
    for i in rs:
        deriv=i.deriv(t,y,par,dct)
        for k in i.reactants:
            for j in i.reactants:
                res[dct[j],dct[k]]=res[dct[j],dct[k]]-deriv[k]
            for j in i.products:
                res[dct[j],dct[k]]=res[dct[j],dct[k]]+deriv[k]
        n+=1
    return res

def kin_solve(par, tlist, y0, rs, specs, dct, vec=vec, jac=jac, NoPlot=True):
    res = solve_ivp(vec, (tlist[0], tlist[-1]), y0=y0, method='Radau',
                    t_eval=tlist, jac=jac, args=(par, rs, specs, dct))
    if not NoPlot:
        plt.figure()
        for i in range(len(specs)):
            plt.plot(res.t, res.y[i], label=specs[i])
        plt.legend()
    return res.y

def frob(par, parnums, names, pops, rcfs, tlist, y0, rs, specs, dct, errs, NoPlot=True, use_ch4_correction=False):
    filtered = [(i, name) for i, name in enumerate(names) if name in dct]
    if not filtered:
        raise ValueError("No overlapping species between data and reaction set")

    inds = [dct[name] for _, name in filtered]
    pops_out = np.array([pops[i] for i, _ in filtered])
    errs_out = np.array([errs[i] for i, _ in filtered])
    name_out = [name for _, name in filtered]

    res = kin_solve(par, tlist, y0, rs, specs, dct, NoPlot=True)
    res_mod = res.copy()

    if use_ch4_correction:
        if 'CH4' in dct and len(par) > len(rs):
            res_mod[dct['CH4']] = y0[dct['CH4']] + (res[dct['CH4']] - y0[dct['CH4']]) * par[len(par)-2]
        if 'C2H6' in dct and len(par) > len(rs)+1:
            res_mod[dct['C2H6']] = res_mod[dct['C2H6']] * par[len(par)-1]

    res_out = res_mod[inds]

    if not NoPlot:
        fig, ax = plt.subplots(len(name_out), 1, sharex=True)
        if len(name_out) == 1:
            ax = [ax]

        for i, name in enumerate(name_out):
            ax[i].plot(tlist, res_out[i], label=f'fit: {name}')
            ax[i].plot(tlist, pops_out[i], '.', label=f'data: {name}')
            ax[i].legend()
        ax[-1].set_xlabel('Time')

    return ((res_out - pops_out) / errs_out).flatten()
