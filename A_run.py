'''Run Ray Tune with Custom Model and Action Masking'''

import os, sys


current_script_dir  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
parent_dir          = os.path.dirname(current_script_dir)         # Get the parent directory (one level up)
sys.path.insert(0, parent_dir)                                    # Add parent directory to sys.path


from D_train import RunRay as train_policy
from B_env import PuzzleGymEnv
from C_policy import CustomMaskedModel as CustomTorchModel



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
            return 0  #train_result['checkpoint_path'] # used for testing

        '''
        # EVALUATE WITH ONE CHECKPOINT ====================================
        def evaluate(self, checkpoint_path=None, train_path= None, test_path = None, max_coalitions_to_plot = 1):
           # Reads from checkpoint and plays policyin env. Graphs Boxplot and Accuracy
           # To load a checkpoint. Num CPU requested should be the at min, same as training. Otherwise look at open issue in Github.
           # :inputs: training dists are needed to initialize the env

           # OBS: to run the sankey diagram, better to do it from a checkpoint as it takes a lot of time

            if test_path is None:
                test_distance_lst=self.test_distance_lst
            else:
                test_distance_lst = self.open_distance_file(filepath = test_path)  # Open and process list of list file

                # Training and testing distances and other vbles for the env
                self.set_config_dict_n_dists(train_path_= train_path, test_path_= test_path)    # sets the self.custom_env_config
                eval_cls = evaluate_policy(checkpoint_path, self.custom_env_config)             # Initialize the Inference class with the checkpoint path and custom environment configuration
                eval_cls.run_inference_and_generate_plots(distance_lst_input=test_distance_lst, max_coalitions=max_coalitions_to_plot) # Run inference on the env and generate coalition plots and 'response_data'
                metrics()                                                                       # Reads the response_data.xls and generates boxplot. Generates the 'summary_table.xls'
        '''




##################################################
# CONFIGS
##################################################

def run_runner(setup_dict, env_config_dict, train_n_eval = True, train_path = None,test_path  = None, checkpoint_path_trained = None):
    '''
    Passess the config variables to RLLIB's trainer
    '''

    #======== Because of the SLURM runner, this needs to be here (otherwise not taken)
    # If we want to use a pre-set list of distances - for reproducibility
    #train_path  = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_training_jan31.txt'
    #test_path  = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_testing_jan31.txt'

    # TRAIN n EVAL
    #train_n_eval = True

    # EVAL
   # train_n_eval = False # inference only
    #checkpoint_path_trained = \
    #"/p/home/jusers/cipolina-kun1/juwels/ray_results/new_distances/PPO_ShapleyEnv_01c47_00000_0_2024-02-01_15-34-29/checkpoint_000290"
    # =====================


    # Use the training distances that worked better - diversified points
    runner = Runner(setup_dict, env_config_dict)

    if train_n_eval:
        # TRAIN (the 'test_path' logic is TODO)
        checkpoint_path_ = runner.train(train_path = train_path, test_path =test_path)

        # EVALUATE
        # NOTE: The 'compute_action' exploration = False gives better results than True
        #runner.evaluate(checkpoint_path = checkpoint_path_,
        #                train_path      = train_path,
        #                test_path       = test_path)

    else: # Evaluate only
      #  runner.evaluate(checkpoint_path = checkpoint_path_trained,
      #                  train_path      = train_path,
      #                  test_path       = test_path
      #                  )
      return




# ==============================================================================================================
# MAIN
# ======================================================================================================================

if __name__ == '__main__':

        env_config_dict = {
                'sides': [5, 6, 7, 8],  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
                'num_pieces': 4,
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