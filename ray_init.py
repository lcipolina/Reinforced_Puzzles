import ray
import time, os
import A_runner

# Ensure all required environment variables are set
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'


# Initialize Ray - connect to the existing cluster
ray.init(address             = 'auto',
         include_dashboard   = False,
         ignore_reinit_error = True,
         log_to_driver       = False,
         _temp_dir           = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp')
# to debug:
#ray.init(local_mode = True,_temp_dir = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp') #>> ray stop (execute before)



# Access SLURM resource variables and set default values if not defined
num_gpus          = int(os.getenv('SLURM_GPUS_PER_TASK', 1))
num_cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
num_workers       = int(os.getenv('SLURM_NTASKS', 1))  # = 'num_tasks' in SLURM (how many parallel processes)

# Resources dict - to pass the resources to the runner (for the trainer)
slurm_config = {
    'num_workers': num_workers,     # how many concurrent tasks (as per SLURM)
    'num_cpus': num_workers,        # For RL, same as num_workers. Otherwise: num_cpus_per_task,  # how many CPUs allocated to each task
    'num_gpus': num_gpus,           # How many GPUs allocated to each task
}


# Call the function from A_runner.py
A_runner.run_runner(slurm_config)

# Shutdown Ray
ray.shutdown()
