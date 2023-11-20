from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

import matplotlib.pyplot as plt

from scipy.special import kv, kvp, gamma
from gptide import cov
from gptide import GPtideScipy
from gptide import mcmc

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


# ------------------------------------- inference -----------------------------------------

# emcee of the inference

def inference_emcee(
    X, U, 
    noise, covparams,
    covfunc, labels,
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
    
