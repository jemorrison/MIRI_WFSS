import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
Run code like so, for example:
> df, popt, popt2, conf_str = jo_fit_trace("GR150R.F090W_A", "dx", "dy", fwcpos_ref=312.123444444)
> write_conffile(conf_str)
"""

def write_conffile(conf_str, conffile=None):
    if conffile is None:
        now = datetime.datetime.now()
        conffile = f"MIRI_WFSS_{now.strftime('%d%b%Y_%H%M')}.conf"
    with open(conffile, 'w') as f:
        for line in conf_str:
            f.write(f"{line}\n")
    print(f"Wrote {conffile}")
    return conffile


def fit_2D26_2(coords, *vars):
    """
    From niriss_specwcs code:
    https://grit.stsci.edu/NIRISS/reference-file-creation/specwcs/-/blob/main/niriss_specwcs/make_specwcs.py?ref_type=heads#L433
    """
    # Generalized 2D polynomial (2,n) order
    (x,y,t) = coords

    #print('in fit 2d26_2', x[0:20], y[0:20], t[0:20])
    e = vars
    n = len(e)//6
    f = 0

    # n = 3 (i 0, 1, 2)

    for i in range(n):
        f = f + t**i * (e[i*6] + x*e[i*6+1] + y*e[i*6+2] + x**2*e[i*6+3] + x*y*e[i*6+4] + y**2*e[i*6+5])
    return f



def jo_fit_trace(tracefile, ind, dep, df=None, fit_torder=2, fit_func=fit_2D26_2):

    if df is None:
        df = pd.read_csv(tracefile, delimiter=" ")
        
    if dep == "dx":
        ind_prefix = "DISPY"
        dep_prefix = "DISPX"
        tind = "dy"
    else:
        ind_prefix  = "DISPX"
        dep_prefix = "DISPY"
        tind = "dx"


    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(df["X0"], df["Y0"])
    plt.show()

    print('In jo_fit_trace', df[dep])
    popt, _ = curve_fit(fit_func, (df["X0"], df["Y0"], df[ind]), # fixed parameters
                            df[dep], # dx or dy (changing values)
                            p0 = np.zeros(6*(fit_torder+1)),
                            ftol=1e-15,
                            xtol=1e-15,
                            maxfev=10000000)
    e = popt
    popt2 = None

    ind1,ind0 = np.min(df[tind]), np.max(df[tind])
    print('ind0, ind1', ind0, ind1)
    def calc_t(ind_val):
        t = ind0 + (ind1 - ind0)*ind_val
        print('t',t)
        return t
    def calc_invt(dep_val):
        ind_val = (dep_val - ind0) / ((ind1-ind0))
        print(ind_val)
        return ind_val

    conf_str = ["NAXIS 1024 1032", "BEAM_+1", "DISPL_+1_0 3.125", "DISPL_+1_1 10.893"]
    ind_conf0 = calc_t(0)
    ind_conf1 = -calc_t(0)+calc_t(1)
    conf_str.append(f"{ind_prefix}_+1_0 {ind_conf0}")
    conf_str.append(f"{ind_prefix}_+1_1 {ind_conf1}")

    order2D = 2
    npolyparams = (order2D+1)*(order2D+2)//2
    nparams = npolyparams * (fit_torder+1)
    n = nparams//npolyparams
    for i in range(n):
        s = f"{dep_prefix}_+1_{i}"
        for ii in range(npolyparams):
            ss = f"{e[i*npolyparams+ii]}"
            s = f"{s} {ss}"
        conf_str.append(s)

    conf_str.append("SENSITIVITY_+1 MIRI_WFSS_+1_sens_pmap0041.fits ")

    return df, popt, popt2, conf_str


def jo_fit_trace_withlambda(tracefile, df=None, fit_torder=2, fit_func=fit_2D26_2, use_grid=False, order2D=2):

    
    if df is None:
        df = pd.read_csv(tracefile, delimiter=" ")

    # convert lambda values to 0 to 1 range
    ts = (df.lam.values-3.25)/(14.01-3.25)
    dxs = df.dx.values
    dys = df.dy.values
    x0s = df.X0.values
    y0s = df.Y0.values

    conf_str = ["NAXIS 1024 1032", "BEAM_+1", "DISPL_+1_0 3.125", "DISPL_+1_1 10.893"]
    
    npolyparams = (order2D+1)*(order2D+2)//2
    nparams = npolyparams * (fit_torder+1)
    n = nparams//npolyparams

   # for xy, param in zip(['X', 'Y'], [dxs, dys]):
   #     popt, pcov = curve_fit(fit_2D26_2, # function we're fitting
   #                            (x0s, y0s, ts), # fixed parameters
   #                            param, # dx or dy (changing values)
   #                            p0 = np.zeros(6*(fit_torder+1)),
   #                            ftol=1e-15,
   #                            xtol=1e-15,
   #                            maxfev=10000000)


    for xy, param in zip(['Y'], [dys]):
        print('CALLING curve_fit for DY')
        print('param is dys')
        popt, pcov = curve_fit(fit_2D26_2, # function we're fitting
                               (x0s, y0s, ts), # fixed parameters
                               param, # dy (changing values)
                               p0 = np.zeros(6*(fit_torder+1)),
                               ftol=1e-15,
                               xtol=1e-15,
                               maxfev=10000000)        
        coeff_start = 0 # indicates which coefficient to start at for different orders


        for i in range(n):
            s = f"DISP{xy}_+1_{i}"
            for ii in range(npolyparams):
                ss = f"{popt[i*npolyparams+ii]}"
                s = f"{s} {ss}"
            conf_str.append(s)

    # redfine things for x

    for xy, param in zip(['X'], [dxs]):
        print('CALLING curve_fit for DX')
        popt, pcov = curve_fit(fit_2D26_2, # function we're fitting
                               (x0s, y0s, ts), # fixed parameters
                               param, # dx 
                               p0 = np.zeros(6*(fit_torder+1)),
                               ftol=1e-15,
                               xtol=1e-15,
                               maxfev=10000000)
        for i in range(n):
            s = f"DISP{xy}_+1_{i}"
            for ii in range(npolyparams):
                ss = f"{popt[i*npolyparams+ii]}"
                s = f"{s} {ss}"
            conf_str.append(s)
            
    conf_str.append("SENSITIVITY_+1 MIRI_WFSS_+1_sens_pmap0041.fits ")
    return df, popt, conf_str



if __name__ == "__main__":

    trace_file = 'trace_grid_output.txt'
    #trace_file = 'trace_ver2.txt' # this only only has values for source 450, 450,

    # Method 1:
    ind = 'dx'
    dep = 'dy'

    #df, popt, popt2, conf_str = jo_fit_trace(trace_file,ind, dep, df=None, fit_torder=2, fit_func=fit_2D26_2)
    df, popt, conf_str = jo_fit_trace_withlambda(trace_file)  

    write_conffile(conf_str)

                                        
