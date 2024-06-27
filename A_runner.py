'''Run Ray Tune with Custom Model and Action Masking'''

# python hierarchical_training.py > rllib_output.log 2>&1


# Tengo que corroborar que este usando bien cada political model - que no se este usando el mismo para todos los agentes

# Cambiar el matching de los lados: "Connected piece 2 side 1 to piece 0 side 1" --> ponerlo con los original side numbers

# EN QUE ESTABA: implementar esto: update_available_connections_n_sides y ver el tema mascaras


import os, sys

os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

CURRENT_DIR  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
parent_dir   = os.path.dirname(CURRENT_DIR)                # Get the parent directory (one level up)
sys.path.insert(0, parent_dir)                             # Add parent directory to sys.path

from E_deploy import Inference as Inference
from D_train import RunRay as train_policy
from Z_utils import get_checkpoint_from_json_file


class Runner:
        def __init__(self, setup_dict, custom_env_config):

           self.setup_dict = setup_dict
           self.custom_env_config = custom_env_config

        # RUN THE ENTIRE TRAINING LOOP ====================================
        def train(self, train_path = None, test_path=None):
            '''Runs the entire training loop
            '''
            # Call Ray to train
            train_result = train_policy(self.setup_dict, self.custom_env_config).train()
            # Reads 'output.xlsx' and generates training graphs avg and rewards per seed - reward, loss, entropy
            # graph_reward_n_others()
            return train_result['checkpoint_path'] # used later for inference


        # EVALUATE WITH ONE CHECKPOINT ====================================
        def evaluate(self, checkpoint_path=None, train_path= None, test_path = None):
            ''' Reads from checkpoint and plays policyin env.'''

            if checkpoint_path is None:
               checkpoint_path = get_checkpoint_from_json_file(directory = os.path.join(CURRENT_DIR, 'results'), name_prefix = 'best_checkpoint', extension = '.json') # Get the latest checkpoint file in the directory by time stamp

            num_eps_to_play = 1
            self.custom_env_config["DEBUG"] = True # Print env at play
            eval_cls = Inference(checkpoint_path, self.custom_env_config, self.setup_dict, num_eps = num_eps_to_play).play_env()             # Initialize the Inference class with the checkpoint path and custom environment configuration
            #eval_cls.run_inference_and_generate_plots(distance_lst_input=test_distance_lst, max_coalitions=max_coalitions_to_plot) # Run inference on the env and generate coalition plots and 'response_data'


##################################################
# CONFIGS
##################################################

def run_runner(setup_dict = None, env_config_dict = None, train_n_eval = True, train_path = None,test_path  = None, checkpoint_path_trained = None):
    '''
    Run RLLIB's trainer and inference classes
    '''

    runner = Runner(setup_dict, env_config_dict)

    if train_n_eval:
        # TRAIN
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

        sides_list = [
        [5, 6, 7, 8],
        [7, 8, 5, 6],
        [5, 6, 7, 8],
        [7, 8, 5, 6]
        ]
        env_config_dict = {
                        'sides': sides_list,  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
                        'num_pieces': len(sides_list),
                        'grid_size': 5,        # 10x10 grid (100 pieces in total)
                        "DEBUG": False,         # Whether to print debug info
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

        print("DONE!")