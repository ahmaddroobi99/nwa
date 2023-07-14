__all__ = ["sunxarray",]

import os

import xarray as xr
import pandas as pd

from . import sunxarray as sunx
from . import myproj

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
area_cp = (122, 124, -15, -13)
area_cp_large = (121, 125, -16, -12)
area_large = (108, 130, -20, -8)
area_very_large = (108, 145, -23, -3)


# ------------------------------ plotting ----------------------------------------------------

# kwargs for maps
mapkw_cp = dict(extent = area_cp, land=None, coastline="10m", figsize=(7,7))
mapkw_cp_large = dict(extent = area_cp_large, land=None, coastline="10m", figsize=(7,7))
mapkw_large = dict(extent = area_large, land="10m", coastline="10m", figsize=(8,5))
mapkw_very_large = dict(extent = area_very_large, land="10m", coastline="10m", figsize=(10,5))

def map_init(zoom, bathy=None, tracks=True, bathy_kw=None, **kwargs):

    if zoom == "cp":
        mapkw = dict(**mapkw_cp)
        clabel=True
        bathy_lvls = [100, 200, 400, 1000]
    elif zoom == "cp_large":
        mapkw = dict(**mapkw_cp_large)
        clabel=True
        bathy_lvls = [100, 200, 400, 1000]
    elif zoom == "large":
        mapkw = dict(mapkw_large)
        clabel=False
        bathy_lvls = [200, 1000]    
    elif zoom == "very_large":
        mapkw = dict(mapkw_very_large)
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
    
def plot_moorings(ax, **kwargs):
    dkwargs = dict(ax=ax, x="longitude", y="latitude", 
                   c="w", s=100, edgecolors="k", marker="*",
                   zorder=30,
                   transform=crs, 
                  )
    dkwargs.update(**kwargs)
    moorings_pd.plot.scatter(**dkwargs)
    
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

def zoom(ds, area):
    """ not sure this is used at the end """
    ds = ds.where(  (ds.xv>=area[0]) & (ds.xv<=area[1]) 
                  & (ds.yv>=area[2]) & (ds.yv<=area[3]), drop=True)
    # !! xarray.where broadcast all variables
    #ds = ds.where(  (ds.xp>=area[0]) & (ds.xp<=area[1]) 
    #              & (ds.yp>=area[2]) & (ds.yp>=area[3]), drop=True)
    return ds

proj = myproj.MyProj("merc")

def project(ds):
    """ fix spatial metrics terms """
    print(ds.suntans.xv)
    ds.suntans.xv, ds.suntans.yv = proj.to_xy(ds.suntans.xv, ds.suntans.yv)
    ds.suntans.xp, ds.suntans.yp = proj.to_xy(ds.suntans.xp, ds.suntans.yp)
    ds.suntans.calc_all_properties()
    # below may not be necessary if performed on all workers
    #ds.suntans.xv, ds.suntans.yv = proj.to_ll(ds.suntans.xv, ds.suntans.yv)
    #ds.suntans.xp, ds.suntans.yp = proj.to_ll(ds.suntans.xp, ds.suntans.yp)

