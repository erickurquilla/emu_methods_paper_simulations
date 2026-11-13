#!/bin/bash
#SBATCH -N 384
#SBATCH -C gpu
#SBATCH -G 1536
#SBATCH -q regular
#SBATCH -J radiating_sphere_test_high_res
#SBATCH --mail-user=eurquill@vols.utk.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -A m3761_g

module load cray-hdf5-parallel/1.14.3.9
module load cudatoolkit/13.2
module load gcc-native/14
module list

echo "shell: SUPPORT=$MPICH_GPU_SUPPORT_ENABLED IPC=$MPICH_GPU_IPC_ENABLED"
export MPICH_GPU_SUPPORT_ENABLED=0
unset MPICH_GPU_IPC_ENABLED
export MPICH_ENV_DISPLAY=1

cd /pscratch/sd/b/bigqke/bigqke_scratch/emu_methods_paper_simulations/bigqke_per/radiating_sphere_test/radiating_sphere_test_L_cm_4.212e6_2.808e6_2.808e6_ncell_384_256_256_dts_9.15e-8_deposition_1_bc_periodic_empty_ppepc_378
srun -n 1536 -G 1536 --gpus-per-task=1 ./main3d.gnu.TPROF.MPI.CUDA.ex job_instructions/inputs

# Submitted batch job 60930041