__all__ = ["sunxarray",]

import os

import xarray as xr
import pandas as pd
import numpy as np

from . import sunxarray as sunx
from . import myproj

proj = myproj.MyProj("merc")

# aurelien's libraries
from mitequinox.utils import load_swot_tracks
import pynsitu as pin
crs = pin.maps.crs


# ------------------------------ general ----------------------------------------------------

suntans_dir = "/home/datawork-lops-osi/aponte/nwa/"

# mooring locations
moorings = dict(
    W310 = [122.8370658081835, -14.13718816307405],
    N280 = [123.02928797854348, -14.052341197573492],
    S245 = [123.03041737634493, -14.230653066337094],
    L245 = [123.03312786502875, -14.230660262826481],
)
moorings_pd = (pd.DataFrame(moorings)
               .T
               .rename(columns={0: "longitude", 1: "latitude"})
)

# areas of interest
bounds = dict(cp=(122, 124, -15, -13),
              cp_large=(121, 125, -16, -12),
              central=(118, 125, -18.5, -13),
              ridge=(120.5, 122, -17.5, -15),
              large=(108, 130, -20, -8),
              very_large=(108, 145, -23, -3),
             )

# ------------------------------ plotting ----------------------------------------------------

# kwargs for maps
mapkw_default = dict(extent = bounds["large"], land="10m", coastline="10m", figsize=(8,5))

def map_init(zoom, bathy=None, tracks=True, bathy_kw=None, **kwargs):

    mapkw = dict(**mapkw_default)
    mapkw["extent"] = bounds[zoom]
    if zoom == "cp":
        mapkw.update(land=None,  figsize=(7,7))
        clabel=True
        bathy_lvls = [100, 200, 400, 1000]
    elif zoom == "cp_large":
        mapkw.update(land=None,  figsize=(7,7))
        clabel=True
        bathy_lvls = [100, 200, 400, 1000]
    elif zoom == "central":
        mapkw.update(figsize=(9,7))
        clabel=False
        bathy_lvls = [100, 200, 1000]
    elif zoom == "ridge":
        mapkw.update(figsize=(5,8))
        clabel=False
        bathy_lvls = [100, 200, 1000]
    elif zoom == "large":
        mapkw.update(figsize=(8,5))
        clabel=False
        bathy_lvls = [200, 1000]    
    elif zoom == "very_large":
        mapkw.update(figsize=(10,5))
        clabel=False
        bathy_lvls = [200, 1000] 
        
    mapkw.update(projection=pin.maps.ccrs.Robinson(central_longitude=120))
    mapkw.update(**kwargs)
    
    fig, ax, _ = pin.maps.plot_map(**mapkw)
    
    if bathy is not None:
        grd, ds = bathy
        bkwargs = dict(clevs=bathy_lvls, filled=False, 
            colors="k", linewidths=0.5,
            colorbar=False, transform=crs,
            zorder=10,
        )
        if bathy_kw is not None:
            bkwargs.update(**bathy_kw)
        _, _, tri, _ = grd.suntans.contourf(ds.dv, **bkwargs)
        if clabel:
            ax.clabel(tri, tri.levels, inline=True, fontsize=10) #fmt=fmt
    
    if tracks:
        plot_swot_tracks(ax)
    
    return fig, ax
    
def plot_moorings(ax, moorings=None, **kwargs):
    dkwargs = dict(c="w", s=100, edgecolors="k", marker="*",
                   zorder=30,
                   transform=crs, 
                  )
    dkwargs.update(**kwargs)
    moorings_pd.plot.scatter(ax=ax, x="longitude", y="latitude", **dkwargs)
    if moorings is not None:
        for p in moorings.Nc:
            m = moorings.sel(Nc=p)
            ax.scatter(m.lonv, m.latv, **dkwargs)
            #ax.text(lon, lat, f"{int(m.Nc)}", size=10, transform=crs, zorder=20)        
    
    
def plot_swot_tracks(ax):
    tracks = load_swot_tracks(bbox=(100, 150, -25, 0))["swath"]
    swot_kwargs = dict(
        facecolor="0.7",
        edgecolor="white",
        alpha=0.5,
        zorder=20,
    )
    #if isinstance(swot_tracks, dict):
    #    swot_kwargs.update(swot_tracks)
    proj = ax.projection
    crs_proj4 = proj.proj4_init
    ax.add_geometries(
        tracks.to_crs(crs_proj4)["geometry"],
        crs=proj,
        **swot_kwargs,
    )
    
def plot_velocity(ax, dsuv, 
                  di=1,
                  uref=1., 
                  xref=.8, yref=.9, # xref=.7, yref=.88
                  **kwargs
                 ):
    """ plot velocity field """
    dkwargs = dict(scale=1e1, add_guide=False)
    dkwargs.update(**kwargs)
    q = (dsuv
         .isel(lon=slice(0,None,di), lat=slice(0,None,di))
         .plot
         .quiver("lon", "lat", "u", "v", ax=ax, transform=crs, **dkwargs)
    )
    ax.quiverkey(q, xref, yref, uref, label=f'{uref}m/s', 
                 labelpos='E', coordinates="figure",
                )

# ------------------------------ suntans -------------------------------------------------

def load_surf():
    zarr = os.path.join(suntans_dir, "suntans_2km_surf")
    ds = xr.open_zarr(zarr)
    #grd = ds[[v for v in ds if "time" not in ds[v].dims]].compute()
    grd = load_grd()
    # switch some variables to coords:
    ds = ds.set_coords(["cells", "dv", "dz", "nfaces", "xp", "xv", "yp", "yv"])
    return ds, grd

def load_grd():
    nc = os.path.join(suntans_dir, "suntans_2km_grid.nc")
    grd = xr.open_dataset(nc)
    return grd

def get_Ac():
    grd = load_grd()
    grd["cells"] = grd.cells.where( grd.cells!=999999, other=-999999 )
    grd.suntans.Nk = np.ones(grd.Nc.size)
    #
    project(grd)
    Ac = (xr.DataArray(grd.suntans.Ac, dims=("Nc"), name="Ac")
          .assign_coords(xv=grd.xv, yv=grd.yv)
          .chunk(dict(Nc=2000))
    )
    return Ac

def load_moorings():
    nc = os.path.join(suntans_dir, "NWS_2km_GLORYS_hex_2013_2014_Nk80dt60_Profile.nc")
    mo = xr.open_dataset(nc)
    mo = mo.set_coords("dz")
    lon, lat = proj.to_ll(mo["xv"], mo["yv"])
    mo = mo.assign_coords(lonv=("Nc", lon), latv=("Nc", lat))
    return mo

def find_point_index(lon, lat, grd):
    """ find index of point closest to lon, lat"""
    d = (grd.xv - lon)**2 + (grd.yv - lat)**2
    return int(d.argmin().compute())

def zoom(ds, area, x="xv", y="yv"):
    """ select points within geographical bounds """
    ds = ds.where(  (ds[x]>=area[0]) & (ds[x]<=area[1]) 
                  & (ds[y]>=area[2]) & (ds[y]<=area[3]), 
                  drop=True,
                 )
    # !! xarray.where broadcast all variables
    #ds = ds.where(  (ds.xp>=area[0]) & (ds.xp<=area[1]) 
    #              & (ds.yp>=area[2]) & (ds.yp>=area[3]), drop=True)
    return ds

def spatial_average(da, Ac, area=None):
    """ spatial average weighted by cell area
    """
    if area is not None:
        da = zoom(da, area)
        Ac = zoom(Ac, area)
    return (da*Ac).sum("Nc")/Ac.sum()

def project(ds):
    """ fix spatial metrics terms """
    #print(ds.suntans.xv)
    ds.suntans.xv, ds.suntans.yv = proj.to_xy(ds.suntans.xv, ds.suntans.yv)
    ds.suntans.xp, ds.suntans.yp = proj.to_xy(ds.suntans.xp, ds.suntans.yp)
    ds.suntans.calc_all_properties()
    # below may not be necessary if performed on all workers
    #ds.suntans.xv, ds.suntans.yv = proj.to_ll(ds.suntans.xv, ds.suntans.yv)
    #ds.suntans.xp, ds.suntans.yp = proj.to_ll(ds.suntans.xp, ds.suntans.yp)

def interpolate_hvelocities(ds, grd, zoom, dx=1e3):
    """ interpolate velocity on a regular grid for plotting purposes"""
    b = bounds[zoom]
    lon, lat = np.arange(b[0], b[1], dx/111e3), np.arange(b[2], b[3], dx/111e3)
    lon2, lat2 = np.meshgrid(lon, lat)
    u = grd.suntans.interpolate(ds.uc.values, lon2, lat2, kind="linear")
    v = grd.suntans.interpolate(ds.vc.values, lon2, lat2, kind="linear")
    dsi = xr.Dataset(dict(u=(("lat", "lon"), u), v=(("lat", "lon"), v)),
                     coords=dict(lon=(("lon",), lon), 
                                 lat=(("lat",), lat),
                                ),
                    )
    return dsi
