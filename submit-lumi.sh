#!/bin/bash -l
#SBATCH --job-name=MarineDebrisUNet
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465001568
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mem=70G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1

# run as:
# cd $HOME/.julia/dev/MarineDebrisUNet
# sbatch submit-lumi.sh --project=. src/litter_classification_train.jl
#
# Assuming the $HOME/julia-depot-marinedebris.tar.xz (with julia and python packages) is present.

VERSION=1.12.0
module load Local-CSC julia/$VERSION julia-amdgpu/1.1.3
module load LUMI/24.03 partition/G
#module load PrgEnv
module load rocm/6.2.2

DEPOT_FILE=$HOME/julia-depot-marinedebris.tar.xz

echo SLURM_JOB_NAME:       $SLURM_JOB_NAME
echo SLURM_JOB_NODELIST:   $SLURM_JOB_NODELIST
echo ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES
echo DEPOT_FILE:           $DEPOT_FILE
echo julia:                $(which julia)
echo arguments:            $@

MY_DEPOT_PATH=/tmp/mkdepot-$USER-$SLURM_JOBID
mkdir -p $MY_DEPOT_PATH

srun flock --nonblock --conflict-exit-code=0 "/tmp/julia-depot-lock-$SLURM_JOBID" tar -xf $DEPOT_FILE -C $MY_DEPOT_PATH

DEPOT=$MY_DEPOT_PATH/julia-depot-marinedebris

export PYTHONPATH="$MY_DEPOT_PATH/julia-depot-marinedebris-py"
export JULIA_DEPOT_PATH="/appl/local/csc/soft/math/julia/$VERSION/share/julia"
export JULIA_DEPOT_PATH="$DEPOT:$JULIA_DEPOT_PATH"
export JULIA_HISTORY="$HOME/.julia/logs/repl_history.jl"

srun julia "$@"
