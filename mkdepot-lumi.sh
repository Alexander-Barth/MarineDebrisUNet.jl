#!/bin/bash -l
#SBATCH --job-name=julia-depot
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=project_465001568
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%j.out

# run as:
# sbatch mkdepot-lumi.sh

# creates the file $HOME/julia-depot-marinedebris.tar.xz containing the necessary julia and python packages

# julia version
VERSION=1.12.0

# load modules
module load Local-CSC julia/$VERSION julia-amdgpu/1.1.3
module load LUMI/24.03 partition/G

# create temporary directories
MY_DEPOT_PATH="/tmp/mkdepot-$USER-$$"
mkdir -p "$MY_DEPOT_PATH/julia-depot-marinedebris"
mkdir -p "$MY_DEPOT_PATH/julia-depot-marinedebris-py"

export JULIA_DEPOT_PATH="$MY_DEPOT_PATH/julia-depot-marinedebris:/appl/local/csc/soft/math/julia/1.12.0/share/julia"
export PYTHON="/appl/local/csc/soft/bio/snakemake/8.4.6/venv/bin/python"
export PYTHONPATH="$MY_DEPOT_PATH/julia-depot-marinedebris-py"
export JULIA_NUM_PRECOMPILE_TASKS="$SLURM_CPUS_PER_TASK"
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# install python package
$PYTHON -m pip install --target "$MY_DEPOT_PATH/julia-depot-marinedebris-py" scikit-learn 

# install julia packages
julia --project=. --eval "using Pkg; Pkg.instantiate()"

# create tar archive
XZ_OPT="--threads=$SLURM_CPUS_PER_TASK"
cd "$MY_DEPOT_PATH"
tar -cvf "$HOME/julia-depot-marinedebris.tar.xz"  julia-depot-marinedebris julia-depot-marinedebris-py
