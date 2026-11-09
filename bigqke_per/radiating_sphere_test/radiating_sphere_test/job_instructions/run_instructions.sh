#!/bin/bash
#SBATCH -N 48
#SBATCH -C gpu
#SBATCH -G 192
#SBATCH -q regular
#SBATCH -J radiating_sphere_test
#SBATCH --mail-user=eurquill@vols.utk.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:20:00
#SBATCH -A m3761_g

module load cray-hdf5-parallel/1.14.3.9
module load cudatoolkit/13.2
module load gcc-native/14
module list

echo "shell: SUPPORT=$MPICH_GPU_SUPPORT_ENABLED IPC=$MPICH_GPU_IPC_ENABLED"
export MPICH_GPU_SUPPORT_ENABLED=0
unset MPICH_GPU_IPC_ENABLED
export MPICH_ENV_DISPLAY=1

cd /pscratch/sd/b/bigqke/bigqke_scratch/emu_methods_paper_simulations/bigqke_per/radiating_sphere_test/radiating_sphere_test
srun -n 192 -G 192 --gpus-per-task=1 ./main3d.gnu.TPROF.MPI.CUDA.ex job_instructions/inputs

# Submitted batch job 60968081