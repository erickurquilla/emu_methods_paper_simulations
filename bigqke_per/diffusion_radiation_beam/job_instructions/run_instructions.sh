#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -q regular
#SBATCH -J diffusion_radiation_beam
#SBATCH --mail-user=eurquill@vols.utk.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:05:00
#SBATCH -A m3761_g

module load cray-hdf5-parallel/1.14.3.9
module load cudatoolkit/13.2
module load gcc-native/14
module list

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0

cd /pscratch/sd/b/bigqke/bigqke_scratch/emu_methods_paper_simulations/bigqke_per/diffusion_radiation_beam
srun -n 8 -G 8 --gpus-per-task=1 ./main3d.gnu.TPROF.MPI.CUDA.ex job_instructions/inputs_multi_energy_isotropic_scattering_test

# Submitted batch job 60318364