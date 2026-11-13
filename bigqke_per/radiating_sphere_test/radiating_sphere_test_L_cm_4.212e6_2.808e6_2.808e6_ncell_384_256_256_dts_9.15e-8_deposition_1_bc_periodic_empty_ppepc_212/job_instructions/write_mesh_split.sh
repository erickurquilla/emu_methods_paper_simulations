#!/bin/bash
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J write_mesh_split_L_cm_4.212e6_2.808e6_2.808e6_ncell_384_256_256_dts_9.15e-8_deposition_1_bc_periodic_empty_ppepc_378
#SBATCH --mail-user=eurquill@vols.utk.edu
#SBATCH --mail-type=ALL
#SBATCH -t 05:00:00
#SBATCH -A m3761

cd /pscratch/sd/b/bigqke/bigqke_scratch/emu_methods_paper_simulations/bigqke_per/radiating_sphere_test/radiating_sphere_test_L_cm_4.212e6_2.808e6_2.808e6_ncell_384_256_256_dts_9.15e-8_deposition_1_bc_periodic_empty_ppepc_378
source /global/common/software/m3761/bigqke_software/python_virtual_enviroments/venv_bigqke_1/bin/activate
python /global/common/software/m3761/bigqke_software/Emu/Scripts/data_reduction/convertToHDF5_split.py

# Submitted batch job 61041730