''' Train a custom env with custom model and action masking on Ray 2.12'''


import os, sys, json, re
import datetime
import numpy as np
import random
import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog

#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
#from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()


current_script_dir  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
parent_dir          = os.path.dirname(current_script_dir)         # Get the parent directory (one level up)
sys.path.insert(0, parent_dir)                                    # Add parent directory to sys.path

from B_env import PuzzleGymEnv as Env                           # Custom environment
from C_policy import CustomMaskedModel as CustomTorchModel  # Custom model

from D_ppo_config import get_sarl_trainer_config            # Tranier config for PPO


output_dir = os.path.expanduser("~/ray_results") # Default output directory
storage_address = "/Users/lucia/Desktop/Art_Project/000-A_Pompei/Repair_project/CODE_puzzle/ray_results/puzzle" #New requirement for Ray 2.12
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")



# Register the custom model
ModelCatalog.register_custom_model("masked_action_model", CustomTorchModel)



#*********************************** RUN RAY TRAINER *****************************************************
class RunRay:


    def __init__(self, setup_dict,custom_env_config, experiment_name = 'puzzle'):
        current_dir            = os.path.dirname(os.path.realpath(__file__))
        self.jason_path  = os.path.join(current_dir, 'best_checkpoint_'+TIMESTAMP+'.json')
        self.clear_json(self.jason_path)

        self.setup_dict        = setup_dict
        self.custom_env_config = custom_env_config
        self.experiment_name   = setup_dict['experiment_name']




    def setup_n_fit(self):
        '''Setup trainer dict and train model
        '''
        #TODO: Set Signal Handler
        #signal.signal(signal.SIGTERM, signal_handler) # Julich SIGTERM handling for graceful shutdown and restore unterminated runs
        experiment_path = os.path.join(output_dir, self.experiment_name)

        #_____________________________________________________________________________________________
        # Setup Config
        #_____________________________________________________________________________________________

        # TRAINER CONFIG - custom model (action and loss) and custom env
        _train_batch_size = self.setup_dict['train_batch_size']
        seed              = self.setup_dict['seed']
        train_iteration   = self.setup_dict['training_iterations']
        num_cpus          = self.setup_dict['cpu_nodes']
        lr_start,lr_end,lr_time = 2.5e-4,  2.5e-5, 50 * 1000000 #embelishments of the lr's

        # Get the trainer with the base configuration  - #OBS: no need to register Env anymore, as it is passed on the trainer config!
        trainer_config = get_sarl_trainer_config(Env, self.custom_env_config,
                            _train_batch_size, lr_start, lr_time, lr_end, num_cpus, seed,
                        )

        #_____________________________________________________________________________________________
        # Setup Trainer
        #_____________________________________________________________________________________________


        #TODO: Set Signal Handler


        tuner = tune.Tuner("PPO", param_space = trainer_config,
                                run_config=air.RunConfig(
                                    name = 'puzzle' , #self.experiment_name,
                                    stop={"training_iteration": train_iteration},
                                    checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50,
                                                                            checkpoint_at_end=True,
                                                                            num_to_keep=3),
                                    local_dir="ray_results",
                                    verbose=2,
                        ))

        result_grid      = tuner.fit() #train the model
        best_result_grid = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
        return best_result_grid


    def train(self):
        ''' Calls Ray to train the model  '''
        if ray.is_initialized(): ray.shutdown()
        ray.init(ignore_reinit_error=True,local_mode=True, storage = storage_address)

        seeds_lst  = self.setup_dict['seeds_lst']
        for _seed in seeds_lst:
            self.set_seeds(_seed)
            print("we're on seed: ", _seed)
            self.setup_dict['seed'] = _seed
            best_res_grid           = self.setup_n_fit()
            self.save_results(best_res_grid,None,self.jason_path, _seed) #print results, saves checkpoints and metrics

        ray.shutdown()
        return 0 #result_dict

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
        best_checkpoint = best_result_grid.checkpoint #returns a folder path, not a file.
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


    def clear_json(self,jason_path):
        with open(jason_path, "w") as f: pass # delete whatever was on the json file