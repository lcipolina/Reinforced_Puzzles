#!/bin/bash
#SBATCH --job-name=ray_puzzle
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --ntasks=31    #31 available in JUSUF batch # How many concurrent tasks. Total: 96 CPUs in Booster , 128 CPUs in Jusuf
#SBATCH --cpus-per-task=1       # How many CPUs each env needs. OBS: Total resources available = ntasks* cpus-per-task
#SBATCH --exclusive               # Exclusive access to the node
#SBATCH --time=02:00:00           # Max time allowed on devel
#SBATCH --output=ray_job_%j.out

# If commented out - will take the default partition
# #SBATCH --partition=dc-cpu-devel #booster #batch # can be left blank #dc-cpu #dc-cpu  in Jureca # develbooster  #booster  #dc-cpu-devel# #batch - default en jusuf
# #SBATCH --mem=64GB # Increase memory allocation -- Not needed and might decrease the number of concurrent tasks



#export RAY_AIR_NEW_OUTPUT=0

# Increase file descriptor limits - makes it work!
ulimit -n 65536

# Set worker registration timeout
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=60

# Disable Ray memory monitor to avoid cgroup issues
export RAY_DISABLE_MEMORY_MONITOR=1

# Set the maximum number of concurrent pending trials
export TUNE_MAX_PENDING_TRIALS_PG=100  # Adjust this number as needed

# Load modules or source your Python environment
#module load Python/3.11

#This didn't work
#source /p/project/ccstdl/cipolina-kun1/reinforced_puzzles/ray_2.2_env/activate.sh
# Activate conda environment


#Experiment
#source /p/scratch/laionize/cache-kun1/miniconda3/bin/activate ray_2.12
source /p/project/ccstdl/cipolina-kun1/reinforced_puzzles/miniconda3/etc/profile.d/conda.sh
conda activate ray_2.12

# Ensure no previous Ray instances are running and Clean up previous Ray session files
ray stop
rm -rf /p/home/jusers/cipolina-kun1/juwels/ray_tmp/*
chmod -R 755 /p/home/jusers/cipolina-kun1/juwels/ray_tmp



# Print the active conda environment to verify
# echo "Active conda environment:"
# conda info --envs
# echo "Current conda environment: $CONDA_DEFAULT_ENV"


# Start Ray head node in the background. Need to provide dir where to find the head IP address
#ray start --head --port=6379 --verbose --temp-dir=/p/home/jusers/cipolina-kun1/juwels/ray_tmp  --dashboard-host 0.0.0.0 &

#ray start --head --port=6379 --temp-dir=/p/home/jusers/cipolina-kun1/juwels/ray_tmp --include-dashboard=False --block

# Test without block
ray start --head --port=6379  --num-cpus=32  --temp-dir=/p/home/jusers/cipolina-kun1/juwels/ray_tmp --include-dashboard=False

# For Jureca and other machines
#/p/scratch/laionize/cache-kun1/ray_env/ray_2.2/venv/bin/python3 /p/scratch/laionize/cache-kun1/ray_env/ray_2.2/venv/bin/ray start --head --port=6379 --verbose --temp-dir=/p/home/jusers/cipolina-kun1/juwels/ray_tmp

# For other machines that can't find the environment
#/p/software/jurecadc/stages/2024/software/Python/3.11.3-GCCcore-12.3.0/bin/python3 /p/scratch/laionize/cache-kun1/ray_env/ray_2.2/venv/bin/ray start --head --port=6379 --verbose --temp-dir=/p/home/jusers/cipolina-kun1/juwels/ray_tmp



# Sleep for a bit to ensure the head node starts properly
sleep 20



# Run your Python script
echo "Calling Python script"
python3 -u /p/project/ccstdl/cipolina-kun1/reinforced_puzzles/ray_init.py

# Stop Ray when done
ray stop

#try with this
# /p/scratch/laionize/cache-kun1/ray_env/ray_2.2/venv/bin/ray stop
