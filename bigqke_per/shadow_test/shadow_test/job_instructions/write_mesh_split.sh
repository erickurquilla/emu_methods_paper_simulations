#!/bin/bash
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J write_mesh_split_shadow_test_L_cm_8.424e6_2.808e6_2.808e6_ncell_192_64_64_dts_3.66e-7
#SBATCH --mail-user=eurquill@vols.utk.edu
#SBATCH --mail-type=ALL
#SBATCH -t 05:00:00
#SBATCH -A m3761

cd /pscratch/sd/b/bigqke/bigqke_scratch/emu_methods_paper_simulations/bigqke_per/shadow_test/shadow_test
source /global/common/software/m3761/bigqke_software/python_virtual_enviroments/venv_bigqke_1/bin/activate
python /global/common/software/m3761/bigqke_software/Emu/Scripts/data_reduction/convertToHDF5_split.py

# Submitted batch job 61471411