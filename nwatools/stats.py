import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

import dask
import dask.array as da

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from scipy.special import kv, kvp, gamma
from gptide import cov
from gptide import GPtideScipy
from gptide import mcmc
import corner
import arviz as az

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

def get_cov_1D(cov_x, cov_t):

    isotropy = False
    if cov_x == "matern12_xy":
        Cx = cov.matern12  # -2 spectral slope
        Cy = cov.matern12  # -2 spectral slope
        C = (Cx, Cy)
    elif cov_x == "matern32_xy":
        Cx = cov.matern32  # -4 spectral slope: not twice differentiable
        Cy = cov.matern32  # -4 spectral slope: 
        C = (Cx, Cy)
    elif cov_x == "matern2_xy":
        #Cov_x = cov.matern_general(np.abs(t_x - t_x.T), 1., 2, λx) # -5 spectral slope
        #Cov_y = cov.matern_general(np.abs(t_y - t_y.T), 1., 2, λy) # -5 spectral slope
        pass
    elif cov_x == "matern52_xy":
        Cx = cov.matern52  # -6 spectral slope
        Cy = cov.matern52  # -6 spectral slope
        C = (Cx, Cy)
    elif cov_x == "expquad":
        #jitter = -10
        #Cx = cov.expquad(t_x, t_x.T, λx) # + 1e-10 * np.eye(Nx)
        #Cy = cov.expquad(t_y, t_y.T, λy) # + 1e-10 * np.eye(Nx)
        assert False, "need revision"

    # isotropic cases
    isotropy = ("iso" in cov_x)
    #if cov_x == "matern2_iso" or True: # dev
    if cov_x == "matern2_iso":
        nu = 2
        #nu = 3/2 # dev
        # for covariances based on distances
        def Cu(x, y, d, λ):
            C = -(
                y**2 * matern_general_d2(d, 1., nu, λ)
                + x**2 * matern_general_d1(d, 1., nu, λ) / d
            )/ d**2
            C[np.isnan(C)] = -matern_general_d2(d[np.isnan(C)], 1.0, nu, λ)
            return C
        def Cv(x, y, d, λ):
            C = -(
                x**2 * matern_general_d2(d, 1., nu, λ)
                + y**2 * matern_general_d1(d, 1., nu, λ) / d
            ) / d**2
            C[np.isnan(C)] = -matern_general_d2(d[np.isnan(C)], 1.0, nu, λ)
            return C
        def Cuv(x, y, d, λ):
            C = x*y*(
                    matern_general_d2(d, 1., nu, λ)
                    - matern_general_d1(d, 1., nu, λ) / d
                ) / d**2
            C[np.isnan(C)] = 0.
            return C
        C = (Cu, Cv, Cuv)
    elif cov_x == "matern32_iso":
        # for covariances based on distances
        def Cu(x, y, d, λ):
            C = -(
                y**2 * matern32_d2(d, 1., λ)
                + x**2 * matern32_d1(d, 1., λ) / d
            )/ d**2
            C[np.isnan(C)] = -matern32_d2(d[np.isnan(C)], 1.0, λ)
            return C
        def Cv(x, y, d, λ):
            C = -(
                x**2 * matern32_d2(d, 1., λ)
                + y**2 * matern32_d1(d, 1., λ) / d
            ) / d**2
            C[np.isnan(C)] = -matern32_d2(d[np.isnan(C)], 1.0, λ)
            return C
        def Cuv(x, y, d, λ):
            C = x*y*(
                    matern32_d2(d, 1., λ)
                    - matern32_d1(d, 1., λ) / d
                ) / d**2
            C[np.isnan(C)] = 0.
            return C
        C = (Cu, Cv, Cuv)
    # dev
    #Cu, Cv, Cuv = (lambda x, y, d, λ: np.eye(*x.shape),)*3 

    Ct = getattr(cov, cov_t)

    return C, Ct, isotropy

def kernel_3d(x, xpr, params, C):
    """
    3D kernel
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            lx: x length scale
            ly: y length scale
            lt: t length scale
            
    """
    eta, lx, ly, lt = params
    Cx, Cy, Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    C *= Cy(x[:,1,None], xpr.T[:,1,None].T, ly) 
    C *= Cx(x[:,0,None], xpr.T[:,0,None].T, lx)
    C *= eta**2
    
    return C

def kernel_3d_iso(x, xpr, params, C):
    """
    3D kernel
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    eta, ld, lt = params
    Cx, Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    d = np.sqrt( (x[:,0,None]  - xpr.T[:,0,None].T)**2 + (x[:,1,None]  - xpr.T[:,1,None].T)**2 )
    C *= Cx(d, ld)
    C *= eta**2
    
    return C

def kernel_3d_iso_uv(x, xpr, params, C):
    """
    3D spatially isotropic kernel, two velocity components
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    eta, ld, lt = params
    Cu, Cv, Cuv, Ct = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    _x = x[:n,0,None] - xpr.T[:n,0,None].T
    _y = x[:n,1,None] - xpr.T[:n,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    #
    C = np.ones((2*n,2*n))
    # test comment out
    C[:n,:n] *= Cu(_x, _y, _d, ld)
    #C[:n,n:] = C[:n,:n] # dev
    #C[n:,:n] = C[:n,:n] # dev
    #C[n:,n:] = C[:n,:n] # dev
    C[:n,n:] *= Cuv(_x, _y, _d, ld)
    C[n:,:n] = C[:n,n:]   # assumes X is indeed duplicated vertically
    C[n:,n:] *= Cv(_x, _y, _d, ld)
    #
    #_Cu  = Cu(_x, _y, _d, ld)
    #_Cv  = Cv(_x, _y, _d, ld)
    #_Cuv  = Cuv(_x, _y, _d, ld)
    #C *= np.block([[_Cu, _Cuv],[_Cuv, _Cv]])
    C *= Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    C *= eta**2
    
    return C

def kernel_3d_iso_uv_traj(x, xpr, params, C):
    """
    3D spatially isotropic kernel, two velocity components
    decorrelate data with different id
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    eta, ld, lt = params
    Cu, Cv, Cuv, Ct = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    _x = x[:n,0,None] - xpr.T[:n,0,None].T
    _y = x[:n,1,None] - xpr.T[:n,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    #
    C = np.ones((2*n,2*n))
    C[:n,:n] *= Cu(_x, _y, _d, ld)
    C[:n,n:] *= Cuv(_x, _y, _d, ld)
    C[n:,:n] = C[:n,n:]   # assumes X is indeed duplicated vertically
    C[n:,n:] *= Cv(_x, _y, _d, ld)
    #
    C *= Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    # decorrelate different trajectories (moorings/drifters)
    C *= ( (x[:,3,None] - xpr.T[:,3,None].T)==0 ).astype(int)
    C *= eta**2
    
    return C

def kernel_3d_iso_u(x, xpr, params, C):
    """
    3D kernel, one velocity component
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    eta, ld, lt = params
    Cu, Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    _x = x[:,0,None] - xpr.T[:,0,None].T
    _y = x[:,1,None] - xpr.T[:,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    C *= Cu(_x, _y, _d, ld)
    C *= eta**2
    
    return C

def kernel_2d_iso_uv(x, xpr, params, C):
    """
    2D kernel (no time), one velocity component
    
    Inputs:
        x: matrices input points [N,2]
        xpr: matrices output points [M,2]
        params: tuple length 2
            eta: standard deviation
            ld: spatial scale
            
    """
    eta, ld = params
    Cu, Cv, Cuv = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    _x = x[:n,0,None] - xpr.T[:n,0,None].T
    _y = x[:n,1,None] - xpr.T[:n,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    #
    C = np.ones((2*n,2*n))
    # test comment out
    C[:n,:n] *= Cu(_x, _y, _d, ld)
    C[:n,n:] *= Cuv(_x, _y, _d, ld)
    C[n:,:n] = C[:n,n:]   # assumes X is indeed duplicated vertically
    C[n:,n:] *= Cv(_x, _y, _d, ld)
    #
    C *= eta**2
    
    return C


def kernel_2d_iso_u(x, xpr, params, C):
    """
    2D kernel, one velocity component
    
    Inputs:
        x: matrices input points [N,2]
        xpr: matrices output points [M,2]
        params: tuple length 2
            eta: standard deviation
            ld: spatial scale
            
    """
    eta, ld = params
    Cu = C
    
    # Build the covariance matrix
    _x = x[:,0,None] - xpr.T[:,0,None].T
    _y = x[:,1,None] - xpr.T[:,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    C = Cu(_x, _y, _d, ld)
    C *= eta**2
    
    return C

def kernel_1d(x, xpr, params, C):
    """
    1D kernel - temporal
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 4
            eta: standard deviation
            lx: x length scale
            ly: y length scale
            lt: t length scale
            
    """
    eta, lt = params
    Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    C *= eta**2
    
    return C

def generate_covariances(model, N, d, λ):
    """ Generate spatial and temporal covariances
    
    """
    
    Cs, Xs, Ns, isotropy = generate_spatial_covariances(
        model[0], N[0], N[1], d[0], d[1], λ[0], λ[1],
    )

    Ct, Xt, Nt = generate_temporal_covariance(
        model[1], N[2], d[2], λ[2],
    )
    
    X = (*Xs, Xt)
    Ns = (*Ns, Nt)
    if isotropy:
        C = (Cs, Ct)
    else:
        C = (*Cs, Ct)

    return C, X, N, isotropy
        
        
def generate_spatial_covariances(model, Nx, Ny, dx, dy, λx, λy):
    """ Generate spatial covariances"""
            
    print(f"Space covariance model: {model}")
    
    N = (Nx, Ny)
    t_x = np.arange(Nx)[:, None]*dx
    t_y = np.arange(Ny)[:, None]*dy

    # space
    Cov_x, Cov_y, Cov_d = (None,)*3
    isotropy=False
    if model == "matern12_xy":
        Cov_x = cov.matern12(t_x, t_x.T, λx) # -2 spectral slope
        Cov_y = cov.matern12(t_y, t_y.T, λy) # -2 spectral slope
    elif model == "matern32_xy":
        Cov_x = cov.matern32(t_x, t_x.T, λx) # -4 spectral slope
        Cov_y = cov.matern32(t_y, t_y.T, λy) # -4 spectral slope
    elif model == "matern2_xy":
        Cov_x = cov.matern_general(np.abs(t_x - t_x.T), 1., 2, λx) # -5 spectral slope
        Cov_y = cov.matern_general(np.abs(t_y - t_y.T), 1., 2, λy) # -5 spectral slope
    elif model == "matern52_xy":
        Cov_x = cov.matern52(t_x, t_x.T, λx) # -6 spectral slope
        Cov_y = cov.matern52(t_y, t_y.T, λy) # -6 spectral slope
    elif model == "expquad":
        jitter = 1e-10
        Cov_x = cov.expquad(t_x, t_x.T, λx) + 1e-10 * np.eye(Nx)
        Cov_y = cov.expquad(t_y, t_y.T, λy) + 1e-10 * np.eye(Nx)
        # relative amplitude of the jitter:
        #    - on first derivative: np.sqrt(jitter) * λx/dx
        #    - on second derivative: np.sqrt(jitter) * (λx/dx)**2
        # with jitter = -10, λx/dx=100, these signatures are respectively: 1e-3 and 1e-1

    C = (Cov_x, Cov_y, Cov_t)

    # for covariances based on horizontal distances
    isotropy = ("iso" in model)
    if isotropy:
        t_x2 = (t_x   + t_y.T*0).ravel()[:, None]
        t_y2 = (t_x*0 + t_y.T  ).ravel()[:, None]
        t_xy = np.sqrt( (t_x2 - t_x2.T)**2 + (t_y2 - t_y2.T)**2 )        
        if model == "matern2_iso":
            Cov_d = cov.matern_general(t_xy, 1., 2, λx) # -5 spectral slope
            C = (Cov_d, Cov_t)
        elif model == "matern32_iso":
            Cov_d = cov.matern32(t_xy, 0., λx) # -4 spectral slope
            C = (Cov_d, Cov_t)
        else:
            assert False, model+" is not implemented"
        
    # Input data points
    xd = (np.arange(0,Nx)[:,None]-1/2)*dx
    yd = (np.arange(0,Ny)[:,None]-1/2)*dy
    X = xd, yd
    
    return C, X, N, isotropy


def generate_temporal_covariance(model, Nt, dt, λt):
    """ Generate temporal covariance"""
    
    print(f"Covariance models: time={model}")
    
    t_t = np.arange(Nt)[:, None]*dt

    if model=="matern12":
        Cov_t = cov.matern12(t_t, t_t.T, λt) # -2 high frequency slope
    elif model=="matern32":
        Cov_t = cov.matern32(t_t, t_t.T, λt) # -4 high frequency slope
        
    # Input data points
    td = (np.arange(0,Nt)[:,None]-1/2)*dt
    
    return Cov_t, td, Nt




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


# ------------------------------------- inference -----------------------------------------

def prepare_inference(
    data_dir, case,
    uv, no_time, parameter_eta_formulation, traj_decorrelation,
):

    # load eulerian flow
    flow_file = os.path.join(data_dir, case+"_flow.zarr")
    dsf = xr.open_dataset(os.path.join(data_dir, flow_file))
    dsf["time"] = dsf["time"]/pd.Timedelta("1D")
    U = dsf.attrs["U"] # useful for computation of alpha much latter

    # problem parameters (need to be consistent with data generation notebook)
    p = dsf.attrs
    for k, v in p.items():
        print(k, v)
    η = p["eta"]  # streamfunction amplitude
    #
    λx = p["lambda_x"]   # km
    λy = p["lambda_y"]   # km
    λt = p["lambda_t"]   # days

    # derived velocity parameter
    γ = η / λx 

    #Lx = float(dsf.x.max()-dsf.x.min())
    #Ly = float(dsf.y.max()-dsf.y.min())
    # km, km this should not matter

    # get 1D covariances
    C, Ct, isotropy = get_cov_1D(p["cov_x"], p["cov_t"])
    Cu, Cv, Cuv = C

    # set covariance parameters
    if no_time:
        if uv:
            covfunc = lambda x, xpr, params: kernel_2d_iso_uv(x, xpr, params, (Cu, Cv, Cuv))
        else:
            covfunc = lambda x, xpr, params: kernel_2d_iso_u(x, xpr, params, Cu)
        covparams = [η, λx]
        labels = ['σ','η','λx',]
    elif isotropy:
        if uv:
            if parameter_eta_formulation:
                covfunc = lambda x, xpr, params: kernel_3d_iso_uv(x, xpr, params, (Cu, Cv, Cuv, Ct))
                covparams = [η, λx, λt]
                labels = ['σ','η','λx','λt']
            else:
                def covfunc(x, xpr, params):
                    # params contains (nu=eta/ld, ld, lt) and needs to be converted to (eta, ld, lt)
                    params = (params[0]*params[1], *params[1:])
                    if traj_decorrelation:
                        return kernel_3d_iso_uv_traj(x, xpr, params, (Cu, Cv, Cuv, Ct))
                    else:
                        return kernel_3d_iso_uv(x, xpr, params, (Cu, Cv, Cuv, Ct))
                covparams = [γ, λx, λt]
                labels = ['σ','γ','λx','λt']
        else:
            covfunc = lambda x, xpr, params: kernel_3d_iso_u(x, xpr, params, (Cu, Ct))
            covparams = [η, λx, λt]
            labels = ['σ','η','λx','λt']
    else:
        covfunc = lambda x, xpr, params: kernel_3d(x, xpr, params, (Cx, Cy, Ct))
        covparams = [η, λx, λy, λt]
        labels = ['σ','η','λx','λy','λt']
    return dsf, covfunc, covparams, labels

## emcee
def inference_emcee(
    X, U, 
    noise, covparams,
    covfunc, labels,
    isotropy=True, no_time=False,
    **kwargs,
):
        
    # Initial guess of the noise and covariance parameters (these can matter)
    if no_time:
        η, λx = covparams
        noise_prior      = gpstats.truncnorm(noise, noise*2, noise/10, noise*10)     # noise
        covparams_priors = [gpstats.truncnorm(η, η*2, η/10, η*10),                   # eta
                            gpstats.truncnorm(λx, λx*2, λx/10, λx*10),               # λx
                           ]
    elif isotropy:
        η, λx, λt = covparams
        noise_prior      = gpstats.truncnorm(noise, noise*2, noise/10, noise*10)     # noise
        covparams_priors = [gpstats.truncnorm(η, η*2, η/10, η*10),                   # eta
                            gpstats.truncnorm(λx, λx*2, λx/10, λx*10),               # λx
                            gpstats.truncnorm(λt, λt*2, λt/10, λt*10),               # λt
                           ]
    else:
        η, λx, λy, λt = covparams
        noise_prior      = gpstats.truncnorm(noise, noise*2, noise/10, noise*10)     # noise
        covparams_priors = [gpstats.truncnorm(η, η*2, η/10, η*10),                   # eta
                            gpstats.truncnorm(λx, λx*2, λx/10, λx*10),               # λx
                            gpstats.truncnorm(λy, λy*2, λy/10, λy*10),               # λy
                            gpstats.truncnorm(λt, λt*2, λt/10, λt*10),               # λt
                           ]

    samples, log_prob, priors_out, sampler = mcmc.mcmc(
        X,
        U,
        covfunc,
        covparams_priors,
        noise_prior,
        nwarmup=100,
        niter=100,
        verbose=False,
    )

    # 40 points
    # with bessels: 2min30, 1.6s / iteration
    # without bessels: 19s, 5 iterations / second
    # with one bessel: 1min05 , 1.5 iteration / second
    # with eye instead of bessel: 20s, 4.7 iterations / second

    # mattern32 - analytical:      24s , 4 iterations per second
    # mattern32 - bessel:      1min32s , 1 iteration  per second

    # 1000 points, mattern32 - analytical: 1 hour total - 30s / iteration

    # should also store prior information
    ds = xr.Dataset(
        dict(samples=(("i", "parameter"), samples),
             priors=(("j", "parameter"), priors_out),
             log_prob=("i", log_prob.squeeze()),
            ),
        coords=dict(parameter=labels)
    )
    ds.attrs["inference"] = "emcee"

    # MAP
    i_map = int(ds["log_prob"].argmax())
    ds["MAP"] = ds["samples"].isel(i=i_map)
    ds.attrs["i_MAP"] = i_map
    #i = np.argmax(log_prob)
    #MAP = samples[i, :]

    return ds

# MH of inference

def inference_MH(
    X, U,
    noise, covparams,
    covfunc, labels,
    n_mcmc = 20_000,
    steps = (1/5, 1/5, 1/5, 1/2),
    tqdm_disable=False,
    **kwargs,
):
    
    η, λx, λt = covparams

    # The order of everything is eta, ld, lt, noise
    #step_sizes = np.array([.5, 5, .5, 0.005])
    #initialisations = np.array([12, 100, 5, 0.01])
    step_sizes = np.array(
        [v*s for v, s in zip([η, λx, λt, noise], steps)]
    )
    initialisations = np.array([η, λx, λt, noise])
    lowers = np.repeat(0, 4)
    #uppers = np.array([100, 1000, 100, 0.05])
    uppers = np.array([η*10, λx*10, λt*10, noise*10])

    # setup objects
    eta_samples = np.empty(n_mcmc)
    ld_samples = np.empty(n_mcmc)
    lt_samples = np.empty(n_mcmc)
    noise_samples = np.empty(n_mcmc)
    accept_samples = np.empty(n_mcmc)
    lp_samples = np.empty(n_mcmc)
    lp_samples[:] = np.nan

    eta_samples[0] = initialisations[0]
    ld_samples[0] = initialisations[1]
    lt_samples[0] = initialisations[2]
    noise_samples[0] = initialisations[3]
    accept_samples[0] = 0

    #covparams_curr = initialisations.copy()[0:3]
    covparams_prop = initialisations.copy()[0:3]    

    # run mcmc
    gp_current = GPtideScipy(X, X, noise, covfunc, covparams)

    for i in tqdm(np.arange(1, n_mcmc), disable=tqdm_disable):

        eta_proposed = np.random.normal(eta_samples[i-1], step_sizes[0], 1)
        ld_proposed = np.random.normal(ld_samples[i-1], step_sizes[1], 1)
        lt_proposed = np.random.normal(lt_samples[i-1], step_sizes[2], 1)
        noise_proposed = np.random.normal(noise_samples[i-1], step_sizes[3], 1)

        proposed = np.array([eta_proposed, ld_proposed, 
                             lt_proposed, noise_proposed])

        if ((proposed.T <= lowers) | (proposed.T >= uppers)).any():
            eta_samples[i] = eta_samples[i-1]
            ld_samples[i] = ld_samples[i-1]
            lt_samples[i] = lt_samples[i-1]
            noise_samples[i] = noise_samples[i-1]
            lp_samples[i] = lp_samples[i-1]
            accept_samples[i] = 0
            continue

        if accept_samples[i-1] == True:
            gp_current = gp_proposed

        covparams_prop = np.array([eta_proposed, ld_proposed, lt_proposed])
        gp_proposed = GPtideScipy(X, X, noise_proposed, covfunc, covparams_prop)

        lp_current = gp_current.log_marg_likelihood(U)
        lp_proposed = gp_proposed.log_marg_likelihood(U)

        alpha = np.min([1, np.exp(lp_proposed - lp_current)])
        u = np.random.uniform()

        if alpha > u:
            eta_samples[i] = eta_proposed
            ld_samples[i] = ld_proposed
            lt_samples[i] = lt_proposed
            noise_samples[i] = noise_proposed
            accept_samples[i] = 1
            lp_samples[i] = lp_proposed
        else:
            eta_samples[i] = eta_samples[i-1]
            ld_samples[i] = ld_samples[i-1]
            lt_samples[i] = lt_samples[i-1]
            noise_samples[i] = noise_samples[i-1]
            accept_samples[i] = 0
            lp_samples[i] = lp_samples[i-1]

    #print(np.mean(accept_samples))

    samples = np.vstack((noise_samples, eta_samples, ld_samples, lt_samples))
    ds = xr.Dataset(
        dict(
            samples=(("i", "parameter"), samples.T), 
            accept=("i", accept_samples),
            log_prob=("i", lp_samples),
            init=(("parameter",), np.roll(initialisations,1)), # need to swap parameter orders
            lower=(("parameter",), np.roll(lowers,1)),
            upper=(("parameter",), np.roll(uppers,1)),
        ),
        coords=dict(parameter=labels)
    )

    #return noise_samples, eta_samples, ld_samples, lt_samples, 
    accepted_fraction = float(ds["accept"].mean())
    print(f"accepted fraction = {accepted_fraction*100:.1f} %")

    # keep only accepted samples
    #ds = ds.where(ds.accept==1, drop=True)

    # MAP
    i_map = int(ds["log_prob"].argmax())
    ds["MAP"] = ds["samples"].isel(i=i_map)

    ds.attrs["accepted_fraction"] = accepted_fraction
    ds.attrs["inference"] = "MH"
    ds.attrs["i_MAP"] = i_map

    return ds

def select_traj_core(ds, Nxy, dx):
    ds = ds.isel(time=0)
    tolerance = .1
    if dx is not None:
        assert Nxy>1, "Nxy must be >1 if dx is specified"
        if Nxy==2:
            # select in a circular shell around the first point
            traj_selection = [np.random.choice(ds.trajectory.values, 1)[0]]
            p0 = ds.sel(trajectory=traj_selection[0])[["x", "y"]]
            d = np.sqrt( (ds["x"]-float(p0.x))**2 + (ds["y"]-float(p0.y))**2 )
            d = d.where( (d>dx*(1-tolerance)) & (d<dx*(1+tolerance)), drop=True )
            if d.trajectory.size==0:
                return
            traj_selection.append(np.random.choice(d.trajectory.values, 1)[0])
        else:
            #Nxy==3: equilateral triangle
            assert False, "Not implemented"
    else:
        traj_selection = np.random.choice(ds.trajectory.values, Nxy, replace=False)
    return traj_selection

def select_traj(*args, repeats=5):
    i, fail = 0, True
    while i<repeats and fail:
        s = select_traj_core(*args)
        if s is not None:
            fail = False
        i+=1
    if fail:
        assert False, "could not select trajectories"
    return s

def mooring_inference(
    dsf, seed,
    covparams, covfunc, labels, N, noise,
    inference="MH", uv=True, no_time=False,
    flow_scale=None, dx=None,
    write=None, overwrite=True,
    **kwargs,
):
    """ run inference for moorings """

    Nt, Nxy = N
    
    if write is not None:
        data_dir, case = write
        output_file = os.path.join(
            data_dir,
            case+f"_moorings_s{seed}_Nxy{Nxy}.nc",
        )
        if flow_scale is not None:
            output_file = output_file.replace(".nc", f"_fs{flow_scale:.2f}.nc")
        if os.path.isfile(output_file) and not overwrite:
            return None
    
    # set random seed - means same mooring positions will be selected across different flow_scales
    np.random.seed(seed)
    
    # randomly select mooring location
    ds = dsf.stack(trajectory=["x", "y"])
    traj_selection = select_traj(ds, Nxy, dx)
    ds = ds.sel(trajectory=traj_selection)
    ds["traj"] = ("trajectory", np.arange(ds.trajectory.size))
    # subsample temporally
    ds = ds.isel(time=np.linspace(0, ds.time.size-1, Nt, dtype=int))

    # set up inference
    if no_time:
        u, v, x, y = xr.broadcast(ds.U, ds.V, ds.x, ds.y)
        assert u.shape==v.shape==x.shape==y.shape
        x = x.values.ravel()
        y = y.values.ravel()
        X = np.hstack([x[:,None], y[:,None],])
    else:
        #u, v, x, y, t = xr.broadcast(ds.U, ds.V, ds.x, ds.y, ds.time)
        u, v, x, y, t, traj = xr.broadcast(ds.U, ds.V, ds.x, ds.y, ds.time, ds.traj)
        assert u.shape==v.shape==x.shape==y.shape==t.shape==traj.shape
        x = x.values.ravel()
        y = y.values.ravel()
        t = t.values.ravel()
        traj = traj.values.ravel()
        #X = np.hstack([x[:,None], y[:,None], t[:,None]])
        X = np.hstack([x[:,None], y[:,None], t[:,None], traj[:,None]])
    u = u.values.ravel()[:, None]
    v = v.values.ravel()[:, None]
    if flow_scale is not None:
        u = u * flow_scale
        v = v * flow_scale
        # update covparams
        covparams = covparams[:]
        covparams[0] = covparams[0] * flow_scale
    # add noise
    u += np.random.randn(*u.shape)*noise
    v += np.random.randn(*v.shape)*noise
    if uv:
        X = np.vstack([X, X])
        U = np.vstack([u, v])
    else:
        U = u        
        
    # reset seed here
    np.random.seed(seed)
    
    # run
    if inference=="MH":
        ds = inference_MH(
            X, U, noise, covparams, covfunc, labels, 
            no_time=no_time, 
            **kwargs,
        )
    elif inference=="emcee":
        ds = inference_emcee(
            X, U, noise, covparams, covfunc, labels, 
            no_time=no_time, 
            **kwargs,
        )
    ds["true_parameters"] = ("parameter", np.array([noise]+covparams))
    ds["seed"] = seed
    if flow_scale is not None:
        ds["flow_scale"] = flow_scale
    
    # store or not and return
    if write is not None:
        ds.to_netcdf(output_file, mode="w")
        return output_file
    else:
        return ds

def run_mooring_ensembles(
    Ne, 
    dsf,
    covparams, covfunc, labels, N, noise,
    step=1/5, **kwargs,
):
    """ wrap mooring_inference to run ensembles """

    dkwargs = dict(tqdm_disable=True, n_mcmc=20_000)
    dkwargs.update(**kwargs)

    # MH default
    dkwargs["steps"] = (step, step, step, 1/2)

    mooring_inference_delayed = dask.delayed(mooring_inference)
    datasets = [
        mooring_inference_delayed(
            dsf, seed, 
            covparams, covfunc, labels, N, noise, 
            **dkwargs,
        ) 
        for seed in range(Ne)
    ]
    datasets = dask.compute(datasets)[0]
    ds = xr.concat(datasets, "ensemble")
    ds = ds.isel(i=slice(0,None,5))
    return ds

def _open_drifter_file(data_dir, case, flow_scale=None):
    
    nc_file = os.path.join(data_dir, case+f"_drifters.nc")
    if flow_scale is not None:
        nc_file = nc_file.replace(".nc", f"_fs{flow_scale:.2f}.nc")
    
    ds = xr.open_dataset(nc_file)
    ds = ds.drop("trajectory").rename_dims(dict(traj="trajectory")) # tmp
    #ds = ds.chunk(dict(trajectory=100, obs=-1))
    #ds = massage_coords(ds)
    ds = ds.rename(lon="x", lat="y")
    ds["x"] = ds["x"]/1e3
    ds["y"] = ds["y"]/1e3    
    ds = ds.assign_coords(t=(ds["time"] - ds["time"][0,0])/pd.Timedelta("1D"))

    # trajectory reaching the end of the simulation
    maxt = ds.time.max("obs")
    n0 = ds.trajectory.size
    ds = ds.where( ~np.isnan(maxt), drop=True)
    ns = ds.trajectory.size
    survival_rate = ns/n0*100
    print(f"{survival_rate:.1f}% of trajectories survived")
    #
    dt = ds.t.differentiate("obs")*day
    ds["u"] = ds.x.differentiate("obs")/dt*1e3 # x are in km
    ds["v"] = ds.y.differentiate("obs")/dt*1e3 # y are in km
    #
    t = ds.t
    #ds = ds.drop(["t", "time"])
    ds = ds.drop(["time"])
    ds["obs"] = ds.t.isel(trajectory=0)
    ds = ds.drop("t").rename(obs="time")    

    return ds, survival_rate

def drifter_inference(
    drifter_dir, case, seed, 
    covparams, covfunc, labels, N, noise,
    inference="MH", uv=True, no_time=False,
    flow_scale=None, dx=None,
    write=None, overwrite=True,    
    **kwargs,
):
    """ run inference for drifters """

    Nt, Nxy = N

    if write is not None:
        data_dir, _ = write
        output_file = os.path.join(
            data_dir,
            case+f"_drifters_s{seed}_Nxy{Nxy}.nc",
        )
        if flow_scale is not None:
            output_file = output_file.replace(".nc", f"_f{flow_scale:.2f}.nc")    
        if os.path.isfile(output_file) and not overwrite:
            return

    # parcels dataset
    ds, survival_rate = _open_drifter_file(drifter_dir, case, flow_scale=flow_scale)

    # set random seed
    np.random.seed(seed)
    
    # randomly select Nxy trajectories
    traj_selection = select_traj(ds, Nxy, dx)
    ds = ds.sel(trajectory=traj_selection)
    ds["traj"] = ("trajectory", np.arange(ds.trajectory.size))

    # subsample temporally
    ds = ds.isel(time=np.linspace(0, ds.time.size-1, Nt, dtype=int))

    # massage inputs to inference problem
    if no_time:
        u, v, x, y = xr.broadcast(ds.u, ds.v, ds.x, ds.y)
        assert u.shape==v.shape==x.shape==y.shape
        x = x.values.ravel()
        y = y.values.ravel()
        X = np.hstack([x[:,None], y[:,None],])
    else:
        #u, v, x, y, t = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time)
        u, v, x, y, t, traj = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time, ds.traj)
        assert u.shape==v.shape==x.shape==y.shape==t.shape==traj.shape
        x = x.values.ravel()
        y = y.values.ravel()
        t = t.values.ravel()
        traj = traj.values.ravel()
        #X = np.hstack([x[:,None], y[:,None], t[:,None]])
        X = np.hstack([x[:,None], y[:,None], t[:,None], traj[:,None]])
    u = u.values.ravel()[:, None]
    v = v.values.ravel()[:, None]
    # add noise
    u += np.random.randn(*u.shape)*noise
    v += np.random.randn(*v.shape)*noise
    if uv:
        X = np.vstack([X, X])
        U = np.vstack([u, v])
    else:
        U = u

    if flow_scale is not None:
        # update covparams
        covparams = covparams[:]
        covparams[0] = covparams[0] * flow_scale
    
    # reset seed here
    np.random.seed(seed)
    
    if inference=="MH":
        ds = inference_MH(
            X, U, noise, covparams, covfunc, labels, 
            no_time=no_time, 
            **kwargs,
        )
    elif inference=="emcee":
        ds = inference_emcee(
            X, U, noise, covparams, covfunc, labels, 
            no_time=no_time, 
            **kwargs,
        )
    ds["true_parameters"] = ("parameter", np.array([noise]+covparams))
    ds["seed"] = seed
    ds["survival_rate"] = survival_rate
    if flow_scale is not None:
        ds["flow_scale"] = flow_scale

    # store or not and return
    if write is not None:
        ds.to_netcdf(output_file, mode="w")
        return output_file
    else:
        return ds
    
def run_drifter_ensembles(
    drifter_dir, case, 
    Ne,
    covparams, covfunc, labels, N, noise,
    step=1/5,
    **kwargs,
):
    """ wrap drifter_inference to run ensembles """

    dkwargs = dict(tqdm_disable=True, n_mcmc=20_000)
    dkwargs.update(**kwargs)

    # MH
    dkwargs["steps"] = (step, step, step, 1/2)

    drifter_inference_delayed = dask.delayed(drifter_inference)
    datasets = [
        drifter_inference_delayed(
            drifter_dir, case, 
            seed,
            covparams, covfunc, labels, N, noise,
            **dkwargs,
        ) 
        for seed in range(Ne)
    ]
    datasets = dask.compute(datasets)[0]
    ds = xr.concat(datasets, "ensemble")
    ds = ds.isel(i=slice(0,None,5))
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

## inference result
def convert_to_az(d, labels, burn=0):
    output = {}
    for ii, ll in enumerate(labels):
        output.update({ll:d[burn:,ii]})
    return az.convert_to_dataset(output)

def plot_inference(ds, stack=False, corner_plot=True, xlim=True, burn=None):

    labels = ds["parameter"].values
    
    #samples = ds.samples.values
    #samples_az = convert_to_az(samples, labels)
    if burn:
        ds = ds.isel(i=slice(burn, None))
    if stack:
        samples = ds.stack(points=("i", "ensemble")).samples.values.T
        samples_az = convert_to_az(samples, labels)
        density_data = [samples_az[labels],]
    else:
        samples = [ds.samples.sel(ensemble=e) for e in ds.ensemble]
        samples_az = [convert_to_az(s, labels) for s in samples]
        density_data = [s[labels] for s in samples_az]
    
    #density_labels = ['posterior',]
    
    axs = az.plot_density(   density_data,
                             shade=0.1,
                             grid=(1, 5),
                             textsize=12,
                             figsize=(15,4),
                             #data_labels=tuple(density_labels),
                             hdi_prob=0.995)

    if "ensemble" in ds.dims:
        cov_params = ds.true_parameters.isel(ensemble=0).values
    else:
        cov_params = ds.true_parameters.values
    
    i=0
    _ds = ds.sel(ensemble=0)
    #for t, ax in zip([noise,]+list(covparams), axs[0]):
    for t, ax in zip(cov_params, axs[0]):
        #print(t, ax)
        ax.axvline(t, color="k", ls="-", label="truth") # true value
        if isinstance(xlim, tuple):
            ax.set_xlim(*xlim)
        elif xlim:
            ax.set_xlim(0, t*2)
        if i==0:
            ax.legend()
        i+=1 
    
    if corner_plot:
        samples = ds.stack(points=("i", "ensemble")).samples.values.T
        fig = corner.corner(
            samples, 
            show_titles=True,
            labels=labels,
            plot_datapoints=True,
            quantiles=[0.16, 0.5, 0.84],
        )
    
def traceplots(ds, MAP=True, burn=None):

    if burn:
        ds = ds.isel(i=slice(burn, None))
    
    fig, axes = plt.subplots(2,2, sharex=True, figsize=(15,6))
    
    for v, ax in zip(ds["parameter"], axes.flatten()[:ds.parameter.size]):
        ds.samples.sel(parameter=v).plot(ax=ax, hue="ensemble", add_legend=False)
        #if MAP:
        #    ax.axvline(ds.attrs["i_MAP"], color="b", lw=2)
        ax.set_title(v.values)
        ax.set_ylabel("")
        #print(v.values)

    fig, ax = plt.subplots(1,1, figsize=(15,3))
    ds["log_prob"].plot(ax=ax, hue="ensemble")
    #if MAP:
    #    ax.axvline(ds.attrs["i_MAP"], color="b", lw=2)
    ax.set_ylabel("")
    ax.set_title("log_prob")



def label_and_print(fig, axs, fig_name):
    """ add labels on figures and print into files """


    if axs is not None:
        for label, ax in axs.items():
            # label physical distance in and down:
            trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    fig_dir = os.path.join(os.getcwd(), "figs")

    for fmt in ["eps", "png"]:
        _fig_name = os.path.join(fig_dir, fig_name+"."+fmt)
        fig.savefig(_fig_name)
        print(f"scp dunree:{_fig_name} .")