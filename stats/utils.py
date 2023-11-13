import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

import matplotlib.pyplot as plt

from scipy.special import kv, kvp, gamma
from gptide import cov

import xrft

day = 86400

# ------------------------------------- covariances ------------------------------------------

# https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

# copy from https://github.com/TIDE-ITRH/gptide/blob/main/gptide/cov.py
def matern_general(dx, eta, nu, l):
    """General Matern base function"""
    
    cff1 = np.sqrt(2*nu)*np.abs(dx)/l
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu)
    K *= np.power(cff1, nu)
    K *= kv(nu, cff1)
    
    K[np.isnan(K)] = np.power(eta, 2.)
    
    return K

# new
def matern_general_d1(dx, eta, nu, l):
    """General Matern base function, first derivative"""
    
    cff0 = np.sqrt(2*nu)/l
    cff1 = cff0*np.abs(dx)
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu) * cff0
    K *= (
        nu*np.power(cff1, nu-1)*kv(nu, cff1)
        + np.power(cff1, nu)*kvp(nu, cff1, n=1)
    )
    K[np.isnan(K)] = 0.
    # but remember K'(d)/d converge toward K''(0) towards 0
    
    return K

def matern_general_d2(dx, eta, nu, l):
    """General Matern base function, second derivative"""
    
    cff0 = np.sqrt(2*nu)/l
    cff1 = cff0*np.abs(dx)
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu) * cff0**2
    K *= (
        nu*(nu-1)*np.power(cff1, nu-2)*kv(nu,cff1) 
        + 2*nu*np.power(cff1, nu-1)*kvp(nu,cff1, n=1)
        + np.power(cff1, nu)*kvp(nu, cff1, n=2)
    )
    K[np.isnan(K)] = -np.power(eta, 2.) * nu/(nu-1)/l**2
    
    return K


def matern32_d1(dx, eta, l):
    """Matern 3/2 function, first derivative"""
    
    cff0 = np.sqrt(3)/l
    cff1 = cff0*np.abs(dx)
    Kp = -np.power(eta, 2.)*cff0*cff1*np.exp(-cff1)
    
    return Kp

def matern32_d2(dx, eta, l):
    """Matern 3/2 function, second derivative"""
    
    cff0 = np.sqrt(3)/l
    cff1 = cff0*np.abs(dx)
    Kpp = np.power(eta, 2.) * cff0**2 *(-1+cff1)*np.exp(-cff1)
    
    return Kpp




def generate_covariances(model, Nx, Ny, Nt, dx, dy, dt, λx, λy, λt):
    
    
    print(f"Covariance models: space={model[0]} , time={model[1]}")
    
    N = (Nx, Ny, Nt)
    t_x = np.arange(Nx)[:, None]*dx
    t_y = np.arange(Ny)[:, None]*dy
    t_t = np.arange(Nt)[:, None]*dt

    # time
    if model[1]=="matern12":
        Cov_t = cov.matern12(t_t, t_t.T, λt) # -2 high frequency slope
    elif model[1]=="matern32":
        Cov_t = cov.matern32(t_t, t_t.T, λt) # -4 high frequency slope

    # space
    Cov_x, Cov_y, Cov_d = (None,)*3
    isotropy=False
    if model[0] == "matern12_xy":
        Cov_x = cov.matern12(t_x, t_x.T, λx) # -2 spectral slope
        Cov_y = cov.matern12(t_y, t_y.T, λy) # -2 spectral slope
    elif model[0] == "matern32_xy":
        Cov_x = cov.matern32(t_x, t_x.T, λx) # -4 spectral slope
        Cov_y = cov.matern32(t_y, _xyt_y.T, λy) # -4 spectral slope
    elif model[0] == "matern2_xy":
        Cov_x = cov.matern_general(np.abs(t_x - t_x.T), 1., 2, λx) # -5 spectral slope
        Cov_y = cov.matern_general(np.abs(t_y - t_y.T), 1., 2, λy) # -5 spectral slope
    elif model[0] == "matern52_xy":
        Cov_x = cov.matern52(t_x, t_x.T, λx) # -6 spectral slope
        Cov_y = cov.matern52(t_y, t_y.T, λy) # -6 spectral slope
    elif model[0] == "expquad":
        jitter = 1e-10
        Cov_x = cov.expquad(t_x, t_x.T, λx) + 1e-10 * np.eye(Nx)
        Cov_y = cov.expquad(t_y, t_y.T, λy) + 1e-10 * np.eye(Nx)
        # relative amplitude of the jitter:
        #    - on first derivative: np.sqrt(jitter) * λx/dx
        #    - on second derivative: np.sqrt(jitter) * (λx/dx)**2
        # with jitter = -10, λx/dx=100, these signatures are respectively: 1e-3 and 1e-1

    C = (Cov_x, Cov_y, Cov_t)

    # for covariances based on horizontal distances
    isotropy = ("iso" in model[0])
    if isotropy:
        t_x2 = (t_x   + t_y.T*0).ravel()[:, None]
        t_y2 = (t_x*0 + t_y.T  ).ravel()[:, None]
        t_xy = np.sqrt( (t_x2 - t_x2.T)**2 + (t_y2 - t_y2.T)**2 )        
        if model[0] == "matern2_iso":
            Cov_d = cov.matern_general(t_xy, 1., 2, λx) # -5 spectral slope
            C = (Cov_d, Cov_t)
        elif model[0] == "matern32_iso":
            Cov_d = cov.matern32(t_xy, 0., λx) # -4 spectral slope
            C = (Cov_d, Cov_t)
        else:
            assert False, model[0]+" is not implemented"
        
    # Input data points
    xd = (np.arange(0,Nx)[:,None]-1/2)*dx
    yd = (np.arange(0,Ny)[:,None]-1/2)*dy
    td = (np.arange(0,Nt)[:,None]-1/2)*dt
    X = xd, yd, td
    
    return C, X, N, isotropy


# ------------------------------------- synthetic field generation --------------------------------------

def generate_uv(kind, N, C, xyt, amplitudes, noise, dask=True, time=True, isotropy=False, seed=1234):
    """ Generate velocity fields
    
    Parameters
    ----------
    kind: str
        "uv": generates u and v independantly
        "pp": generates psi (streamfunction) and (phi) independantly from which velocities are derived
    N: tuple
        Grid dimension, e.g.: (Nx, Ny, Nt)
    C: tuple
        Covariance arrays, e.g.: (Cov_x, Cov_y, Cov_t)
    xyt: tuple
        xd, yd, td coordinates
    amplitudes: tuple
        amplitudes (u/v or psi, phi) as a size two tuple
    dask: boolean, optional
        activate dask distribution
    time: boolean, optional
        activate generation of time series
    isotropy: boolean, optional
        horizontally isotropic formulation
    seed: int, optional
        random number generation seed
    """
    
    # prepare output dataset
    xd, yd, td = xyt
    ds = xr.Dataset(
        coords=dict(x=("x", xd[:,0]), y=("y", yd[:,0]), time=("time", td[:,0])),
    )
    ds["time"].attrs["units"] = "days"
    ds["x"].attrs["units"] = "km"
    ds["y"].attrs["units"] = "km"
    if not time:
        ds = ds.drop("time")
        
    # perform Cholesky decompositions
    if isotropy:
        Cov_x, Cov_t = C
    else:
        Cov_x, Cov_y, Cov_t = C
    Lx = np.linalg.cholesky(Cov_x)
    Lt = np.linalg.cholesky(Cov_t)
    if not isotropy:
        Ly = np.linalg.cholesky(Cov_y)
    # start converting to dask arrays
    t_chunk = 5
    #Lt_dask = da.from_array(Lt).persist()
    if dask:
        Lx = da.from_array(Lx, chunks=(-1, -1))
        Lt = da.from_array(Lt, chunks=(t_chunk, -1)).persist()

    # generate sample
    u0_noise, u1_noise = 0., 0.
    rstate = da.random.RandomState(seed)
    np.random.seed(seed)
    if time and not isotropy:
        if dask:
            U0 = rstate.normal(0, 1, size=N, chunks=(-1, -1, t_chunk))
            U1 = rstate.normal(0, 1, size=N, chunks=(-1, -1, t_chunk))
            # noise
            if noise>0:
                u0_noise = noise * rstate.normal(0, 1, size=N, chunks=(-1, -1, t_chunk))
                u1_noise = noise * rstate.normal(0, 1, size=N, chunks=(-1, -1, t_chunk))
    elif not time and not isotropy:
        U0 = np.random.normal(0, 1, size=(N[0], N[1]))
        U1 = np.random.normal(0, 1, size=(N[0], N[1]))
        # noise
        if noise>0:
            u0_noise = noise * np.random.normal(0, 1, size=(N[0], N[1]))
            u1_noise = noise * np.random.normal(0, 1, size=(N[0], N[1]))
    elif time and isotropy:
        if dask:
            _N = (N[0]*N[1], N[2])
            U0 = rstate.normal(0, 1, size=_N, chunks=(-1, t_chunk))
            U1 = rstate.normal(0, 1, size=_N, chunks=(-1, t_chunk))
            # noise
            if noise>0:
                u0_noise = noise * rstate.normal(0, 1, size=_N, chunks=(-1, t_chunk))
                u1_noise = noise * rstate.normal(0, 1, size=_N, chunks=(-1, t_chunk))
    elif not time and isotropy:
        U0 = np.random.normal(0, 1, size=(N[0]*N[1],))
        U1 = np.random.normal(0, 1, size=(N[0]*N[1],))
        # noise
        if noise>0:
            u0_noise = noise * np.random.normal(0, 1, size=(N[0]*N[1],))
            u1_noise = noise * np.random.normal(0, 1, size=(N[0]*N[1],))
    
    
    #return Lx, Lt, U0, u0_noise
    # 2D
    #zg = η * Lx @ V @ Lt.T
    # 3D
    # with dask
    #zg = η * da.einsum("ij,kl,mn,jln->ikm", Lx, Ly, Lt_dask, V)
    # with opt_einsum
    import opt_einsum as oe
    if time and not isotropy:
        u0 = amplitudes[0] * oe.contract("ij,kl,mn,jln", Lx, Ly, Lt, U0)
        u1 = amplitudes[1] * oe.contract("ij,kl,mn,jln", Lx, Ly, Lt, U1)
    elif not time and not isotropy:
        u0 = amplitudes[0] * oe.contract("ij,kl,jl", Lx, Ly, U0)
        u1 = amplitudes[1] * oe.contract("ij,kl,jl", Lx, Ly, U1)
    elif time and isotropy:
        u0 = amplitudes[0] * oe.contract("ij,kl,jl", Lx, Lt, U0)
        u1 = amplitudes[1] * oe.contract("ij,kl,jl", Lx, Lt, U1)            
    elif not time and isotropy:
        u0 = amplitudes[0] * Lx @ U0
        u1 = amplitudes[1] * Lx @ U1
            
    # add noise
    u0 = u0 + u0_noise
    u1 = u1 + u1_noise
    
    _u0 = u0
    
    if isotropy and time:
        # final reshaping required
        u0 = u0.reshape(N)
        u1 = u1.reshape(N)
    elif isotropy and not time:
        # final reshaping required
        u0 = u0.reshape(N[0], N[1])
        u1 = u1.reshape(N[0], N[1])
        
    #return _u0, u0
    
    if time:
        dims = ("x", "y", "time")
    else:
        dims = ("x", "y",)
    if kind=="uv":
        ds["U"] = (dims, u0)
        ds["V"] = (dims, u1)
    elif kind=="pp":
        ds["psi"] = (dims, u0)
        ds["phi"] = (dims, u1)
        # rederive u
        dpsidx = ds.psi.differentiate("x")
        dpsidy = ds.psi.differentiate("y")
        dphidx = ds.phi.differentiate("x")
        dphidy = ds.phi.differentiate("y")
        ds["U"] = -dpsidy + dphidx
        ds["V"] =  dpsidx + dphidy

    ds = ds.transpose(*reversed(dims))
    ds.attrs.update(kind=kind)

    return ds


# ------------------------------------- plotting ------------------------------------------

def plot_snapshot(ds, i=None, **kwargs):

    if i is not None:
        ds = ds.isel(time=i)
    
    if "psi" in ds:
        return plot_snapshot_pp(ds, **kwargs)
    

def plot_snapshot_pp(ds, darrow=20):
    
    fig, axes = plt.subplots(3,2,figsize=(15,15), sharex=True)
    
    dsa = ds.isel(x=slice(0,None,darrow), y=slice(0,None,darrow))

    ax = axes[0, 0]
    ds.psi.plot(ax=ax, cmap="RdBu_r")
    dsa.plot.quiver("x", "y", "U", "V", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("psi")

    ax = axes[0, 1]
    ds.phi.plot(ax=ax, cmap="RdBu_r")
    dsa.plot.quiver("x", "y", "U", "V", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("phi")
    
    ##
    ax = axes[1, 0]
    ds.U.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "U", "V", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("u")

    ax = axes[1, 1]
    ds.V.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "U", "V", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("v")
    
    ##
    divergence = ds.U.differentiate("x")/1e3 + ds.V.differentiate("y")/1e3
    vorticity = ds.V.differentiate("x")/1e3 - ds.U.differentiate("y")/1e3
    
    ax = axes[2, 0]
    divergence.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "U", "V", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("divergence")

    ax = axes[2, 1]
    vorticity.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "U", "V", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("vorticity")    
    
    return fig, axes



def plot_spectra(ds, v, yref=1e-1, slopes=[-4,-5,-6], **kwargs):
    
    # compute spectra
    dkwargs = dict(dim=['x','y'], detrend='linear', window=True)
    E = xrft.power_spectrum(ds[v], **kwargs)
    E = E.compute()
    E_iso = xrft.isotropic_power_spectrum(ds[v], truncate=True, **dkwargs)
    print(E_iso)
    if "time" in E.dims:
        E = E.mean("time")
        E_iso = E_iso.mean("time")
    E = E.compute()    
    E_iso = E_iso.compute()    
    
    # plot in kx-ky space
    _E = E.where( (E.freq_x>0) & (E.freq_y>0), drop=True )
    fig, ax = plt.subplots(1,1)
    np.log10(_E).plot(**kwargs)
    np.log10(_E).plot.contour(levels=[-8, -4, 0], colors="w", linestyles="-")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{v}: kx-ky power spectrum")
    
    # plot isotropic
    #fy = 1e-3
    #_Ex = E.sel(freq_y=fy, method="nearest")
    #_Ex = _Ex.where(_Ex.freq_x>0, drop=True)

    fig, ax = plt.subplots(1,1)
    E_iso.plot(label="iso", lw=4, zorder=10)
    #_Ex.plot(label=f"E(f_y={fy:.1e})")
    #np.log10(_E).plot.contour(levels=[-8, -4, 0], colors="w", linestyles="-")

    _f = np.logspace(-2.5, min(-.5, float(np.log10(E_iso.freq_r.max()))), 10)
    for s in slopes:
        ax.plot(_f, yref * (_f/_f[0])**s, color="k")
        ax.text(_f[-1], yref * (_f[-1]/_f[0])**s, r"$f^{}$".format(int(s)))
    #ax.plot(_f, yref * (_f/_f[0])**-4, color="k")
    #ax.text(_f[-1], yref * (_f[-1]/_f[0])**-4, r"$f^{-3}$")
    #ax.plot(_f, yref * (_f/_f[0])**-6, color="k")
    #ax.text(_f[-1], yref * (_f[-1]/_f[0])**-6, r"$f^{-6}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"{v}: isotropic spectrum")
    
