# --------------------- application related librairies and parameters ------------------

import sys
from subprocess import check_output, STDOUT

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime
from matplotlib import pyplot as plt

import pynsitu as pin
crs = pin.maps.crs

import nwatools as nwa

from dask import delayed
import threading
#from tqdm import tqdm
#from tqdm.contrib.logging import logging_redirect_tqdm

# ---- Run parameters

# movie parameters
v, cmap, clims = "vorticity", 'RdBu_r', (-2, 2)
v, cmap, clims = "temp", pin.thermal, (27, 28)
v, cmap, clims = "temp", "gnuplot2", (22, 33)   # large, year

zoom = "cp_large"
zoom = "large"

start, end = "2013/09/02T00:00", "2013/09/04T00:00"
start, end = None, None # year long movie

name = "nwa_"+zoom+"_"+v

fig_dir = "/home1/scratch/aponte/figs"

# dask parameters
#dask_jobs = 2
dask_jobs = 10
jobqueuekw = dict(processes=7, cores=7)

# ---------------------------- dask utils - do not touch -------------------------------

import os, shutil
import logging

import dask
from dask.distributed import Client, LocalCluster
from dask.delayed import delayed
from dask.distributed import performance_report, wait
from distributed.diagnostics import MemorySampler

# to update eventually to avoid dependency
import mitequinox.utils as ut

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def spin_up_cluster(jobs):
    logging.info("start spinning up dask cluster, jobs={}".format(jobs))
    cluster, client = ut.spin_up_cluster(
        "distributed",
        jobs=jobs,
        fraction=0.9,
        walltime="12:00:00",
        **jobqueuekw,
    )
    
    logging.info("dashboard via ssh: " + ut.dashboard_ssh_forward(client))
    return cluster, client


def close_dask(cluster, client):
    
    logging.info("starts closing dask cluster ...")

    try:    
        client.close()
        logging.info("client closed ...")
        # manually kill pbs jobs
        manual_kill_jobs()
        logging.info("manually killed jobs ...")
        # cluster.close()
        # logging.info("cluster closed ...")
    except:
        logging.exception("cluster.close failed ...")
        # manually kill pbs jobs
        manual_kill_jobs()

    logging.info("... done")


def manual_kill_jobs():
    """manually kill dask pbs jobs"""

    import subprocess, getpass

    #
    username = getpass.getuser()
    #
    bashCommand = "qstat"
    try:
        output = subprocess.check_output(bashCommand, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    #
    for line in output.splitlines():
        lined = line.decode("UTF-8")
        if username in lined and "dask" in lined:
            pid = lined.split(".")[0]
            bashCommand = "qdel " + str(pid)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # print(e.output.decode())
                pass

def dask_compute_batch(computations, client, batch_size=None):
    """ breaks down a list of computations into batches
    """
    # compute batch size according to number of workers
    if batch_size is None:
        # batch_size = len(client.scheduler_info()["workers"])
        batch_size = sum(list(client.nthreads().values()))
    # find batch indices
    total_range = range(len(computations))
    splits = max(1, np.ceil(len(total_range)/batch_size))
    batches = np.array_split(total_range, splits)
    # launch computations
    outputs = []
    for b in batches:
        logging.info("batches: " + str(b)+ " / "+str(total_range))
        out = dask.compute(*computations[slice(b[0], b[-1]+1)])
        outputs.append(out)
        
        # try to manually clean up memory
        # https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
        client.run(gc.collect)
        client.run(trim_memory)  # should not be done systematically
        
    return sum(outputs, ())



# ---------------------------------- core of the job to be done ---------------------------------- 

def clean_fig_dir():
    """ delete all figures """
    import os, shutil
    folder = fig_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def generate_mpg(name, framerate=24):
    """ generate mpg files
    Requires ffmpeg in environment, do with image2Movies otherwise
    https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg
    """
    # -y overwrites existing file
    com = f'''ffmpeg -y -framerate {framerate} -pattern_type glob -i '{fig_dir}/*.png' -c:v libx264 -pix_fmt yuv420p /home1/scratch/aponte/{name}.mp4'''
    output = check_output(com, shell=True, stderr=STDOUT, universal_newlines=True).rstrip('\n')

if __name__ == "__main__":

    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed.log",
        level=logging.INFO,
        #level=logging.DEBUG,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions
    
    
    logging.info("0 - collect input variables and clean up fig_dir")
    
    # python movies.py zoom start end
    #name = sys.argv[1]
    #zoom = sys.argv[2]
    #start, end = sys.argv[3], sys.argv[4]
    #logging.info("argv : "+" / ".join(sys.argv))
    logging.info(f" Movie name: {name}")
    logging.info(f" Movie start/end times: {start} - {end}")
    logging.info(f" Movie geographical zoom: {zoom}")
    
    clean_fig_dir()
    logging.info(" all files in fig_dir deleted : "+fig_dir)
    
    # spin up cluster
    logging.info("1 - spin up dask cluster")
    # distributed
    cluster, client = spin_up_cluster(dask_jobs)
    # local
    #cluster = LocalCluster()
    #client = Client(cluster)

    # create tiling
    logging.info("2 - load data")
    ds, grd = nwa.load_surf()
    
    if start is None and end is not None:    
        ds = ds.sel(time=slice(start, end))
    
    # add deformations
    if v in ["vorticity", "divergence"]:
        zarr = os.path.join(nwa.suntans_dir, f"suntans_2km_surf_gradients")
        dsd = xr.open_zarr(zarr)
        if start is None and end is not None:
            dsd = dsd.sel(time=slice(start, end))
        f = pin.geo.coriolis(dsd.yv)
        ds["vorticity"] = (dsd["dvcdx"] - dsd["ducdy"])/f
        #ds["divergence"] = dsd["ducdx"] + dsd["dvcdy"]/f
        #ds["strain_normal"] = dsd["ducdx"] - dsd["dvcdy"]
        #ds["strain_shear"] = dsd["dvcdx"] + dsd["ducdy"]
        ds[v] = dsd[v]
    #
    ds = ds[[v]].persist() # only keep variable of interest at the moment
    wait(ds)
    
    #
    logging.info("3 - start main loop")
    
    def print_figure(da, i):

        MPL_LOCK = threading.Lock()
        with MPL_LOCK:

            plt.switch_backend('agg')

            fig, ax = nwa.map_init(zoom, bathy=(grd, da))
            _, _, poly, cbar = grd.suntans.plotcelldata(da, 
                                                        vmin=clims[0], vmax=clims[1], 
                                                        cmap=cmap, 
                                                        crs=crs,
                                                       )

            t = str(da.time.dt.strftime('%Y/%m/%d %Hh').values)
            ax.set_title(t)

            fig_file = os.path.join(fig_dir, f"{i:05d}.png")
            fig.savefig(fig_file, dpi=150)
            plt.close(fig)
        
    print_figure_delayed = delayed(print_figure)        
    
    n_t = ds.time.size
    n_workers = len(client.scheduler_info()['workers'])
    #with logging_redirect_tqdm():    
    #for i in tqdm(range(0, n_t, n_workers)):
    for i in range(0, n_t, n_workers):
        values = [print_figure_delayed(ds[v].isel(time=j), j) 
                  for j in range(i, min(i+n_workers, n_t))]    
        futures = client.compute(values)
        results = client.gather(futures)
        logging.info(f" {i}-{i+n_workers-1} done ")

    # close dask
    close_dask(cluster, client)
    
    # produce movie
    logging.info("4 - make movie")
    generate_mpg(name)
    logging.info(f" movie should be ready: /home1/scratch/aponte/{name}.mp4'")
    logging.info(f" scp -p dunree:/home1/scratch/aponte/{name}.mp4 .")
    
    logging.info("- all done")
