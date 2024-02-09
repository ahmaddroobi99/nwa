#!/usr/bin/env python
# coding: utf-8

# # 3D (x-y-t) field inference: ensemble experiments
# 
# We start from a flow field generated with `drifter_3D.ipynb` and run multiple inferences with different draws of  mooring positions & drifters release positions as well as noise.

import os
import logging
from time import sleep, time

import xarray as xr

import nwatools.stats as st
day = 86400

#data_dir = "data/"
#data_dir = "/home1/scratch/aponte/"
data_dir = "/home/datawork-lops-osi/aponte/nwa/drifter_stats"

# actually run production inferences
prod=True
#prod=False

# flow case
U = "0.1"
case = "3D_matern32_iso_matern12_pp_r0.0_u"+U

uv = True # True if u and v are observed
parameter_eta_formulation = False # eta vs gamma formulation
noise = 0.01 # observation noise added to u/v
no_time = False # activates time inference
no_space = False # activates time inference
traj_decorrelation = False  # artificially decorrelate different moorings/drifters
#traj_decorrelation = True  # artificially decorrelate different moorings/drifters
if no_space:
    # makes little sense otherwise
    traj_decorrelation = True
assert not no_time, "need to implement decorrelation across time"

# number of points used for inference#
#Nxy, Nt = 1, 50
#Nxy, Nt = 2, 50
Nxy, Nt = 4, 50
#Nxy, Nt = 8, 50
#Nxy, Nt = 16, 50
#Nxy, Nt = 5, 50
#Nxy, Nt = 10, 50

# run multiple Nxy at once
Nxy = [1, 2, 4, 8, 16]
Nxy = [2, 4, 8, 16]

# number of ensembles
#Ne = 10 # dev
Ne = 100 # prod

dx = None
#dx = 100. # Nxy>1, separation between platforms
#dx = [20, 50, 100, 150, 200]

# dask parameters
dask_jobs = 5  # number of dask pbd jobs
jobqueuekw = dict(processes=20, cores=20)  # uplet debug

# ---------------------------------- main ------------------------------------

def run():

    logging.info("starting run")

    # ### prepare inference & common utils

    dsf, covfunc, covparams, labels = st.prepare_inference(
        data_dir, case,
        uv, no_time, no_space,
        parameter_eta_formulation, traj_decorrelation,
    )

    # ---
    # ## mooring inference
    logging.info("starting mooring inference ...")

    def run_mooring(Nxy, dx):
        if dx is not None:
            logging.info(f" Nxy= {Nxy}, dx={dx} - start")
        else:
            logging.info(f" Nxy= {Nxy} - start")
        
        # build output file name
        nc = os.path.join(data_dir, case+f"_moorings_ensemble_Nxy{Nxy}.nc")
        if dx is not None:
            nc = nc.replace(".nc", f"_dx{dx:0.0f}.nc")
        if traj_decorrelation:
            nc = nc.replace(".nc", f"_trajd.nc")

        if prod and not os.path.isfile(nc):
            ds = st.run_mooring_ensembles(
                Ne, dsf, covparams, covfunc, labels, (Nt, Nxy), noise, dx=dx,
                no_time=no_time, no_space=no_space,                
            ) 
            ds.to_netcdf(nc, mode="w")
        else:
            logging.info(" ... skipping")
            ds = xr.open_dataset(nc)
            ds["parameter"] = ds.parameter.astype(str)
        logging.info(f" Nxy= {Nxy}, dx={dx} - end")
        return ds

    if isinstance(dx, list):
        Dm = []
        for d in dx:
            Dm.append(run_mooring(Nxy, d))
        ds = Dm[0]
    elif isinstance(Nxy, list):
        Dm = []
        for n in Nxy:
            Dm.append(run_mooring(n, dx))
        ds = Dm[0]
    else:
        ds = run_mooring(Nxy, dx)

    logging.info(" ... ending mooring inference")


    # ---
    # ## drifter inference
    logging.info("starting drifter inference ...")

    def run_drifter(Nxy, dx):
        logging.info(f" Nxy= {Nxy}, dx={dx} - start")

        # build output file name    
        nc = os.path.join(data_dir, case+f"_drifters_ensemble_Nxy{Nxy}.nc")
        if dx is not None:
            nc = nc.replace(".nc", f"_dx{dx:0.0f}.nc")
        if traj_decorrelation:
            nc = nc.replace(".nc", f"_trajd.nc")

        if prod and not os.path.isfile(nc):
            ds = st.run_drifter_ensembles(
                data_dir, case, Ne, covparams, covfunc, labels, (Nt, Nxy), noise, dx=dx,
                no_time=no_time, no_space=no_space,
            ) 
            ds.to_netcdf(nc, mode="w")
        else:
            logging.info(" ... skipping")
            ds = xr.open_dataset(nc)
            ds["parameter"] = ds.parameter.astype(str)
        logging.info(f" Nxy= {Nxy}, dx={dx} - end")
        return ds

    if isinstance(dx, list):
        Dr = []
        for d in dx:
            Dr.append(run_drifter(Nxy, d))
        ds = Dr[0]
    elif isinstance(Nxy, list):
        Dr = []
        for n in Nxy:
            Dr.append(run_drifter(n, dx))
        ds = Dr[0]
    else:
        ds = run_drifter(Nxy, dx)    

    logging.info(" ... ending drifter inference")


# ---------------------------------- dask ------------------------------------

def spin_up_cluster(
    ctype,
    jobs=None,
    processes=None,
    fraction=0.8,
    timeout=20,
    **kwargs,
):
    """Spin up a dask cluster ... or not
    Waits for workers to be up for distributed ones

    Paramaters
    ----------
    ctype: None, str
        Type of cluster: None=no cluster, "local", "distributed"
    jobs: int, optional
        Number of PBS jobs
    processes: int, optional
        Number of processes per job (overrides default in .config/dask/jobqueue.yml)
    timeout: int
        Timeout in minutes is cluster does not spins up.
        Default is 20 minutes
    fraction: float, optional
        Waits for fraction of workers to be up

    """

    if ctype is None:
        return
    elif ctype == "local":
        from dask.distributed import Client, LocalCluster

        dkwargs = dict(n_workers=14, threads_per_worker=1)
        dkwargs.update(**kwargs)
        cluster = LocalCluster(**dkwargs)  # these may not be hardcoded
        client = Client(cluster)
    elif ctype == "distributed":
        from dask_jobqueue import PBSCluster
        from dask.distributed import Client

        assert jobs, "you need to specify a number of dask-queue jobs"
        cluster = PBSCluster(processes=processes, **kwargs)
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        if not processes:
            processes = cluster.worker_spec[0]["options"]["processes"]

        flag = True
        start = time()
        while flag:
            wk = client.scheduler_info()["workers"]
            logging.info("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk) >= processes * jobs * fraction:
                flag = False
                logging.info("Cluster is up, proceeding with computations")
            now = time()
            if (now - start) / 60 > timeout:
                flag = False
                logging.info("Timeout: cluster did not spin up, closing")
                cluster.close()
                client.close()
                cluster, client = None, None

    return cluster, client


def dashboard_ssh_forward(client):
    """returns the command to execute on a local computer in order to
    have access to dashboard at the following address in a browser:
    http://localhost:8787/status
    """
    env = os.environ
    port = client.scheduler_info()["services"]["dashboard"]
    return f'ssh -N -L {port}:{env["HOSTNAME"]}:8787 {env["USER"]}@datarmor1-10g', port


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
        output = subprocess.check_output(
            bashCommand, shell=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
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
                # logging.info(e.output.decode())
                pass

# ---------------------------------- main ------------------------------------

if __name__=="__main__":


    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed.log",
        level=logging.INFO,
        # level=logging.DEBUG,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions


    # spin up cluster
    logging.info("start spinning up dask cluster, jobs={}".format(dask_jobs))
    cluster, client = spin_up_cluster(
        "distributed",
        jobs=dask_jobs,
        fraction=0.9,
        walltime="12:00:00",
        **jobqueuekw,
    )
    ssh_command, dashboard_port = dashboard_ssh_forward(client)
    logging.info("dashboard via ssh: " + ssh_command)
    logging.info(f"open browser at address of the type: http://localhost:{dashboard_port}")

    # main run
    run()

    # close dask
    close_dask(cluster, client)
