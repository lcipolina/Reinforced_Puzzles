''' Train a custom env with custom model and action masking on Ray 2.12'''


import os, sys, json, re
import datetime
import numpy as np
import random
import ray
from ray import air, tune
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
from sigterm_handler import signal_handler, return_state_file_path          # Resume after SIGTERM termination

current_script_dir  = os.path.dirname(os.path.realpath(__file__))           # Get the current script directory path
parent_dir          = os.path.dirname(current_script_dir)                   # Get the parent directory (one level up)
sys.path.insert(0, parent_dir)                                              # Add parent directory to sys.path

from B_env_hrl import PuzzleGymEnv as Env                                   # Custom environment
from D_ppo_config import get_marl_hrl_trainer_config as get_trainer_config  # Trainer config for single agent PPO

output_dir = os.path.expanduser("~/ray_results") # Default output directory
TIMESTAMP  = datetime.datetime.now().strftime("%Y%m%d-%H%M")


#*********************************** RUN RAY TRAINER *****************************************************
class RunRay:

    def __init__(self, setup_dict,custom_env_config):
        current_dir            = os.path.dirname(os.path.realpath(__file__))
        self.jason_path        = os.path.join(current_dir, 'results', 'best_checkpoint_'+TIMESTAMP+'.json')
        self.setup_dict        = setup_dict
        self.custom_env_config = custom_env_config
        self.experiment_name   = setup_dict.get('experiment_name', 'puzzle')

    def setup_n_fit(self):
        '''Setup trainer dict and train model
        '''

        #_____________________________________________________________________________________________
        # Setup Config
        #_____________________________________________________________________________________________

        train_iteration   = self.setup_dict['training_iterations']
        lr_start,lr_end,lr_time = 2.5e-4,  2.5e-5, 50 * 1000000 #embelishments of the lr's

        # Get the trainer with the base configuration  - #OBS: no need to register Env anymore, as it is passed on the trainer config!
        trainer_config = get_trainer_config(Env, self.custom_env_config, self.setup_dict,
                            lr_start, lr_time, lr_end )

        #_____________________________________________________________________________________________
        # Setup Trainer with SLURM signal handler
        #_____________________________________________________________________________________________

        # SLURM signal handler - Decide whether to start a new experiment or restore an existing one
        experiment_path = os.path.join(output_dir, self.experiment_name)
        state_file_path = return_state_file_path()
        if os.path.exists(state_file_path):  #restore works only for unterminated runs
            with open(state_file_path, "r") as f:
                state = f.read().strip()
            if state == "interrupted":
                print("Previous run was interrupted. Attempting to restore...")
                tuner = tune.Tuner.restore(
                path=experiment_path,
                trainable="PPO",
                resume_unfinished=True,
                resume_errored=False,
                restart_errored=False,
                param_space=trainer_config,  # Assuming `trainer_config` matches the original setup
            )
            os.remove(state_file_path) # Clear the state file after handling

        else: # Train from scratch
            print("Starting a new experiment run.")
            tuner  = tune.Tuner("PPO",
                    param_space = trainer_config,
                    run_config = air.RunConfig(
                        name =  self.experiment_name,
                        stop = {"training_iteration": train_iteration}, # "iteration" will be the metric used for reporting
                        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50,
                                                               checkpoint_at_end=True,
                                                               num_to_keep= 3 ),#keep only the last 3 checkpoints
                        #callbacks = [wandb_callbacks],  # WandB local_mode = False only!
                        verbose= 3, #0 for less output while training - 3 for seeing custom_metrics better
                        local_dir = output_dir
                            )
                        )

        result_grid      = tuner.fit() #train the model
        best_result_grid = result_grid.get_best_result(metric="episode_reward_mean", mode="max")

        print("\n=== BEST RESULT ===\n")
        print(f" Reward_max: {best_result_grid.metrics['sampler_results']['episode_reward_max']}")
        print(f" Reward_mean: {best_result_grid.metrics['sampler_results']['episode_reward_mean']}")

        return best_result_grid

    #____________________________________________________________________________________________
    #  Train the model
    #____________________________________________________________________________________________
    def train(self):
        ''' Calls Ray to train the model  '''
        if ray.is_initialized(): ray.shutdown()
        ray.init(ignore_reinit_error=True,local_mode=False)

        seeds_lst  = self.setup_dict['seeds_lst']
        for _seed in seeds_lst:
            self.set_seeds(_seed)
            print("we're on seed: ", _seed)
            self.setup_dict['seed'] = _seed
            best_res_grid           = self.setup_n_fit()
            result_dict             = self.save_results(best_res_grid,None,self.jason_path, _seed) #print results, saves checkpoints and metrics

        ray.shutdown()
        return result_dict  # checkpoint path

    #____________________________________________________________________________________________
    #  Analize results and save files
    #____________________________________________________________________________________________

    def save_results(self, best_result_grid, excel_path, json_path, _seed):
        '''Save results to Excel file and save best checkpoint to JSON file
           :input: best_result_grid is supposed to bring the best iteration, but then we recover the entire history to plot
        '''

        # Process results
        df = best_result_grid.metrics_dataframe  #Access the entire *history* of reported metrics from a Result as a pd DataFrame. And not just the best iteration

        # Save best checkpoint (i.e. the last) onto JSON filer
        best_checkpoints = []
        best_checkpoint = best_result_grid.checkpoint  #returns a folder path, not a file.
        path_checkpoint = best_checkpoint.path
        checkpoint_path = path_checkpoint if path_checkpoint else None
        best_checkpoints.append({"seed": _seed, "best_checkpoint": checkpoint_path})
        with open(json_path, "a") as f:  # Save checkpoints to file
            json.dump(best_checkpoints, f, indent=4)

        return {'checkpoint_path': checkpoint_path}



    #____________________________________________________________________________________________
    # Aux functions
    #____________________________________________________________________________________________

    def set_seeds(self,seed):
        torch.manual_seed(seed)           # Sets seed for PyTorch RNG
        torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
        np.random.seed(seed=seed)         # Set seed for NumPy RNG
        random.seed(seed)                 # Set seed for Python's random RNG
