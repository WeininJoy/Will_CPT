#!/bin/bash
#!
#! SLURM job script for Tursa (DiRAC Extreme Scaling, EPCC)
#! GPU nodes used as CPU-only: 32 cores (A100-40) or 48 cores (A100-80)
#! Hybrid MPI + OpenMP: mpi4py.futures workers + OpenMP ODE solver
#!

#! sbatch directives begin here ###############################
#SBATCH -J integer_allowedK_DE
#! !! CHANGE: replace with your Tursa project account !!
#SBATCH -A YOUR_TURSA_ACCOUNT
#! !! CHANGE: verify partition name with `sinfo` on Tursa login node !!
#! Common options: gpu, cpu — check `sinfo` to confirm
#SBATCH -p gpu
#SBATCH --nodes=1
#! 1 master process + 4 worker processes = 5 MPI tasks
#! Sized for the smaller A100-40 nodes (32 cores): 5 * 6 = 30 cores <= 32
#! If on A100-80 nodes (48 cores) you can increase to --ntasks=7 --cpus-per-task=6
#SBATCH --ntasks=5
#! Each MPI task gets 6 CPUs for OpenMP threads
#SBATCH --cpus-per-task=6
#! Do not allocate GPUs (CPU-only job on GPU node)
#SBATCH --gres=gpu:0
#SBATCH --time=36:00:00
#SBATCH --mail-type=NONE

#! sbatch directives end here #################################

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e 's/^\([0-9][0-9]*\).*$/\1/')

#! Load modules — Tursa uses OpenMPI, not Intel MPI
#! !! CHANGE: verify exact module names with `module avail` on Tursa !!
module purge
module load gcc/12.2.0
module load openmpi/4.1.5-gcc12-cpu

#! !! CHANGE: update path to your virtual environment on Tursa's filesystem !!
source /path/to/your/tursa/venv/bin/activate
which python

#! Each MPI worker uses OMP_NUM_THREADS OpenMP threads for compute_U_matrices()
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

#! mpi4py.futures: launch with mpirun -np N python -m mpi4py.futures script.py
np=$SLURM_NTASKS
application="python -m mpi4py.futures find_intK_baye_opt_planck_bounds.py"
options=""

CMD="mpirun -np $np $application $options"

#! !! CHANGE: update to the path of this script on Tursa !!
workdir="/path/to/your/tursa/diff_evol"

###############################################################
cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
deactivate
