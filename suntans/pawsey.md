# run jupyterlab and dask cluster on pawsey

Matt's doc: https://github.com/mrayson/pawsey-containers/tree/master/hpc-python/jupyter-sfoda

```
cd /software/projects/pawsey0106/aponte # shortcut: cdw
sbatch start_jupyter.slm .
# monitor job
squeue -u aponte # shortcut: qs
# monitor jupyter job output for ssh tunnel command and grab ssh tunnel command:
tail -f jupyter-???.out
```

In a local shell windown, execute ssh tunnel command:

```
ssh -N -l ...
```

Open jupyterlab in browser: http://localhost:8888
Open dashboard in browser: http://localhost:8787/status

Generate two terminals in the jupyter browser and in each:
1. launch scheduler:
```
cd /software/projects/pawsey0106/aponte
dask-scheduler --scheduler-file scheduler-$HOSTNAME.json --idle-timeout 0
# you can now link notebook to scheduler
```
2. launch workers
```
ssh localhost "cd /software/projects/pawsey0106/aponte/ && sbatch --ntasks=128 -c 2 --time=04:00:00 start_worker.slm scheduler-$HOSTNAME.json ./"
#ssh localhost "cd /software/projects/pawsey0106/aponte/ && sbatch --ntasks=32 -c 8 --time=10:00:00 start_worker.slm scheduler-$HOSTNAME.json ./"
# configuration set in startup_jupyter.slm
```

Dask configuration: 8 cores, 16GB (see startup_jupyter.slm)


After computation is over, kill scheduler:
```
scancel ...
```

ps x | grep "ssh"
#kill notebook
