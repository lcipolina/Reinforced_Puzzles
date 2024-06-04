#!/bin/bash
#SBATCH --job-name=ray_puzzle
#SBATCH --account=cstdl
#SBATCH --partition=booster  #batch # devel
#SBATCH --nodes=1
#SBATCH --ntasks=40  #96 CPUs in Booster
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00  #aparently max time allowed on devel
#SBATCH --output=ray_job_%j.out

export RAY_AIR_NEW_OUTPUT=0


# Load modules or source your Python environment
module load CUDA/12

source /p/home/jusers/cipolina-kun1/juwels/vision_env/sc_venv_template/activate.sh
#source /p/home/jusers/cipolina-kun1/juwels/miniconda3/etc/profile.d/conda.sh
#conda activate ray_2.6

# Start the Ray head node
ray start --head --port=6379 --block &

#Include dashboard
#ray start --head --port=6379 --block --verbose --temp-dir=/p/fastdata/mmlaion/cipolina/ --dashboard-host 0.0.0.0 &

# Sleep for a bit to ensure the head node starts properly
sleep 10

# Run your Python script
python -u /p/home/jusers/cipolina-kun1/juwels/coalitions/ray_init.py

# Stop Ray when done
ray stop
