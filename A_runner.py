'''Run Ray Tune with Custom Model and Action Masking'''

import os, sys


CURRENT_DIR  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
parent_dir   = os.path.dirname(CURRENT_DIR)                # Get the parent directory (one level up)
sys.path.insert(0, parent_dir)                             # Add parent directory to sys.path

from E_deploy import Inference as inference
from D_train import RunRay as train_policy
from Z_utils import get_checkpoint_from_json_file




class Runner:
        def __init__(self, setup_dict, custom_env_config):

           self.setup_dict = setup_dict
           self.custom_env_config = custom_env_config

        # RUN THE ENTIRE TRAINING LOOP ====================================
        def train(self,train_path = None,test_path=None):
            '''Runs the entire training loop
            '''
            # Call Ray to train
            train_result         = train_policy(self.setup_dict, self.custom_env_config).train()
            # Reads 'output.xlsx' and generates training graphs avg and rewards per seed - reward, loss, entropy
            #graph_reward_n_others()
            return train_result['checkpoint_path'] # used later for inference


        # EVALUATE WITH ONE CHECKPOINT ====================================
        def evaluate(self, checkpoint_path=None, train_path= None, test_path = None):
           # Reads from checkpoint and plays policyin env. Graphs Boxplot and Accuracy
           # :inputs: training dists are needed to initialize the env

            if checkpoint_path is None:
               checkpoint_path = get_checkpoint_from_json_file(directory = os.path.join(CURRENT_DIR, 'results'), name_prefix = 'best_checkpoint', extension = '.json') # Get the latest checkpoint file in the directory by time stamp

            num_eps_to_play = 2
            eval_cls = inference(checkpoint_path, self.custom_env_config, num_eps = num_eps_to_play).play_env()             # Initialize the Inference class with the checkpoint path and custom environment configuration
            #eval_cls.run_inference_and_generate_plots(distance_lst_input=test_distance_lst, max_coalitions=max_coalitions_to_plot) # Run inference on the env and generate coalition plots and 'response_data'


##################################################
# CONFIGS
##################################################

def run_runner(slurm_config = None,setup_dict = None, env_config_dict = None, train_n_eval = True, train_path = None,test_path  = None, checkpoint_path_trained = None):
    '''
    Passess the config variables to RLLIB's trainer
    slurm_config: dict with the number of workers, cpus and gpus - coming from 'ray_init.py via call on the SLURM run
    '''

    #======== Because of the SLURM runner, this needs to be here (otherwise not taken)
    # If we want to use a pre-set list of distances - for reproducibility
    #train_path  = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_training_jan31.txt'
    #test_path  = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_testing_jan31.txt'

    #if setup_dict is None or env_config_dict is None:
    env_config_dict = {
                    'sides': [5, 6, 7, 8],  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
                    'num_pieces': 16,
                    "DEBUG": False,         # Whether to print debug info
                    }

    setup_dict = { 'training_iterations': 2,
                    'train_batch_size': 600,
                    'seeds_lst': [42],
                    'cpu_nodes': slurm_config.get('num_cpus', 7),
                    'experiment_name': 'puzzle',
                    }

    # TRAIN n EVAL
    train_n_eval = True

    # EVAL
   # train_n_eval = False # inference only
    #checkpoint_path_evaluate = \
    #"/p/home/jusers/cipolina-kun1/juwels/ray_results/new_distances/PPO_ShapleyEnv_01c47_00000_0_2024-02-01_15-34-29/checkpoint_000290"
    # =====================


    # Use the training distances that worked better - diversified points
    runner = Runner(setup_dict, env_config_dict)

    if train_n_eval:
        # TRAIN (the 'test_path' logic is TODO)
        checkpoint_path_trained = runner.train(train_path = train_path, test_path =test_path)

        # EVALUATE
        # NOTE: The 'compute_action' exploration = False gives better results than True
        runner.evaluate(checkpoint_path = checkpoint_path_trained,
                        train_path      = train_path,
                        test_path       = test_path)

    else: # Evaluate only
        checkpoint_path_evaluate = None  # It will take last checkpoint on file
        runner.evaluate(checkpoint_path = checkpoint_path_evaluate,
                        train_path      = train_path,
                        test_path       = test_path
                        )
    return 0




# ==============================================================================================================
# MAIN
# ======================================================================================================================

if __name__ == '__main__':

        # For SLURM these need to be inside a function

        env_config_dict = {
                'sides': [5, 6, 7, 8],  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
                'num_pieces': 6,
                "DEBUG": True         # Whether to print debug info
                }

        setup_dict = { 'training_iterations': 2,
                        'train_batch_size': 600,
                        'seeds_lst': [42],
                        'cpu_nodes': 7,
                        'experiment_name': 'puzzle',
                        }


        # TRAIN n Inference
        train_n_eval = True
        checkpoint_path_trained = None
        train_path, test_path = None, None

        # EVAL
        #train_n_eval = False # inference only
        # checkpoint_path_trained = \
        # "/Users/lucia/ray_results/subadditive_test/PPO_DynamicCoalitionsEnv_4ec75_00000_0_2024-05-26_15-53-18/checkpoint_000004"
        # =====================

        run_runner(setup_dict,
                   env_config_dict,
                   train_n_eval,
                   train_path = train_path,
                   test_path  = test_path,
                    checkpoint_path_trained =checkpoint_path_trained )