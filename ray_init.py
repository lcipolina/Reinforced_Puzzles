#!/usr/bin/env python3

import os
import subprocess
import ray

def initialize_ray():

    # Initialize Ray - connect to the existing cluster
    ray.init(
        address='auto',
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False
    )

    print("Ray initialized")


def run_script():
    import A_runner

    # Access SLURM resource variables and set default values if not defined
    num_gpus = int(os.getenv('SLURM_GPUS_PER_TASK', 1))
    num_cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    num_workers = int(os.getenv('SLURM_NTASKS', 1))  # = 'num_tasks' in SLURM (how many parallel processes)

    # Resources dict - to pass the resources to the runner (for the trainer)
    slurm_config = {
        'num_workers': num_workers,  # how many concurrent tasks (as per SLURM)
        'num_cpus': num_workers,  # For RL, same as num_workers. Otherwise: num_cpus_per_task,  # how many CPUs allocated to each task
        'num_gpus': num_gpus,  # How many GPUs allocated to each task
    }

    print('calling the runner')
    # Call the function from A_runner.py
    A_runner.run_runner(slurm_config)


def main():

    print("Entering ray_init")

    # Ensure Ray uses only CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        initialize_ray()
        run_script()
    except Exception as e:
        print(f"Ray initialization or script execution failed: {e}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    # Activate the conda environment within the script if not already activated
    if not os.getenv('CONDA_PREFIX'):
        activate_conda = "/p/project/ccstdl/cipolina-kun1/reinforced_puzzles/miniconda3/bin/activate"
        conda_env = "ray_2.12"
        command = f"source {activate_conda} {conda_env} && python3 -u /p/project/ccstdl/cipolina-kun1/reinforced_puzzles/ray_init.py"
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.wait()
    else:
        main()
