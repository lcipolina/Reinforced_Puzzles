'''Load the saved model and evaluate it in the environment

'''

import torch, socket, os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info

from Z_utils import my_print
from B_env_naive import PuzzleGymEnv as Env                       # custom environment
from C_policy import CustomMaskedModel as CustomTorchModel        # Custom model with masks
from D_ppo_config import get_sarl_trainer_config                  # Configuration for the training


#current_script_dir  = os.path.dirname(os.path.realpath(__file__))
#parent_dir          = os.path.dirname(current_script_dir) # Get the parent directory (one level up)
#sys.path.insert(0, parent_dir) # Add parent directory to sys.path

current_dir = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
# To save results in Juelich
juelich_dir = '/p/scratch/ccstdl/cipolina-kun/'
# Define paths
home_dir = '/Users/lucia/ray_results'
hostname = socket.gethostname() # Determine the correct path
output_dir = juelich_dir if 'jwlogin21.juwels' in hostname.lower() else home_dir
os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = output_dir

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")





class Inference:

    def __init__(self, checkpoint_path, custom_env_config, setup_dict, num_eps = 1):
        self.checkpoint_path   = checkpoint_path
        self.custom_env_config = custom_env_config
        self.setup_dict        = setup_dict
        self.num_episodes_during_inference = num_eps                     # How many episodes to run during inference
        self.env               = Env(custom_env_config)
        register_env("custom_env", lambda env_ctx: self.env)                         # the register_env needs a callable/iterable
        ModelCatalog.register_custom_model("masked_action_model", CustomTorchModel)  # Register the custom model - used by D_ppo_config.py


    #==================================================================
    # Step 1 - Run the environment and collect responses
    #==================================================================

    def play_env(self):

        '''Play the environment with the trained model and generate responses and final coalitions
        '''
        if ray.is_initialized(): ray.shutdown()
        ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
        #ray.init(address='auto',include_dashboard=False, ignore_reinit_error=True,log_to_driver=True, _temp_dir = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp')

        # Rebuild the policy
        #algo = Algorithm.from_checkpoint(self.checkpoint_path)

        #     Rebuild the policy to work with fewer number of workers
        # def policy_dict():
        #     return {f"policy{i}": PolicySpec(observation_space=self.env.observation_space,
        #                              action_space=self.env.action_space) for i in self.env._agent_ids}
        # def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        #     '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
        #     get_policy_key = lambda agent_id: f"policy{agent_id}"
        #     return get_policy_key(agent_id)


        # Convert checkpoint info to algorithm state
        checkpoint_info = get_checkpoint_info(self.checkpoint_path)
        state = Algorithm._checkpoint_info_to_algorithm_state(
            checkpoint_info=checkpoint_info,
            policy_ids=None,  # Adjust if using specific policy IDs; might be auto-inferred in your case
            policy_mapping_fn=None,  # Set to None as we'll configure policies directly in the config
            policies_to_train=None,  # Specify if training specific policies
        )

        # Need to bring the config dict exactly as it was when training (otherwise won't work)
        self.setup_dict['cpu_nodes'] = 1  # Modify the configuration for your multi-agent setup and 1 CPU cores
        modified_config  = get_sarl_trainer_config(Env, self.custom_env_config, self.setup_dict)
        state["config"]  = modified_config.to_dict()     # Inject the modified configuration into the state
        algo = Algorithm.from_state(state)              # Load the algorithm from the modified state


        #==================================================
        # Run the environment on the trained model
        #==================================================
        obs, _ = self.env.reset()

        num_episodes = 0
        episode_reward = 0.0
        while num_episodes < self.num_episodes_during_inference:

            # Compute SARL action (with default value for policy)
            action = algo.compute_single_action(observation=obs, explore=False, policy_id="default_policy") # Pre-process and Flattens the obs inside
            obs, reward, terminated, _, _ = self.env.step(action)
            my_print(f"Reward: {reward}, Done: {terminated}",DEBUG=True)
            episode_reward += reward

           # self.env.render()          # Convert current puzzle into string for visualization

            if terminated:
                my_print("The puzzle has been solved or the episode is over!", DEBUG = True)
                my_print(f"Episode done: Total reward = {episode_reward}", DEBUG=True)
                obs, _ = self.env.reset()
                num_episodes += 1
                episode_reward = 0.0
                break

            # Compute MARL action
            # agent_id = list(obs_dict.keys())[0]
            # policy_agent = "policy" + str(agent_id)
            # action, states, extras_dict = algo.get_policy(policy_id=policy_agent).compute_single_action(obs_dict, explore=False)
            # response_entry = {'agent_id': agent_id, 'action': action, 'rew': reward_current_agent}

            # Save results

        ray.shutdown()
        return 0

    #=======================================================================================
    # Step 2 - Saving and plotting results
    #=======================================================================================

    #def save_results_to_excel(self, responses_by_distance, accepted_coalitions_by_distance):
    #    '''Save the responses and final coalitions to an Excel file'''
    #    responses_df = self.convert_responses_to_dataframe(responses_by_distance)
    #    coalitions_df = self.convert_coalitions_to_dataframe(accepted_coalitions_by_distance)
    #    with pd.ExcelWriter(response_file) as writer:
    #        responses_df.to_excel(writer, sheet_name='Responses')

    # def convert_responses_to_dataframe(self, responses_by_distance):
    #     '''Convert the responses dictionary to a pandas DataFrame'''
    #     data = []
    #     for distance_str, responses in responses.items():
    #         for response in responses:
    #             data.append({
    #                 'agent_id': response['agent_id'],  #read with this label by the 'metrics' script
    #                 'Action'  : response['action'],
    #                 'rew'     : response['rew']             #read with this label by the 'metrics' script
    #             })
    #     return pd.DataFrame(data)


    #================================================
    # RUN INFERENCE AND GENERATE PLOTS
    #================================================
    # def run_inference_and_generate_plots(self, distance_lst_input=None,max_coalitions=None):
    #     '''Run inference on the environment and generate coalition plots and 'response_data'
    #        :distance_lst_input: is a list of distance lists to be used in the environment
    #        :max_coalitions: is the maximum number of coalitions to be generated for each agent'''

    #     if max_coalitions is not None: # Trim the distance list to make this more manageable
    #         distance_lst_input = distance_lst_input[:max_coalitions]

    #     responses_by_distance, accepted_coalitions_by_distance = self.play_env_custom_obs(distance_lst_input) # The first return stores the responses of all agents and the second stores the accepted coalitions
    #     self.plot_final_coalitions(responses_by_distance, accepted_coalitions_by_distance)
    #     #self.sankey_diagram(responses_by_distance) # takes forever to save if there are multiple nodes.
    #     self.save_results_to_excel(responses_by_distance, accepted_coalitions_by_distance)  # Assuming this method exists for saving results to Excel



###############################################
# MAIN
################################################
if __name__=='__main__':


    # SETUP for env and model --> MAKE SURE IT MATCHES THE TRAINING CODE !!


    custom_env_config = {
                'num_agents'     : 4,
                'max_steps'      : 8000,
                 'batch_size'    : 1000 # for the CV learning - one update per batch size
                }

    setup_dict = {'num_cpus':7}
   # cls = Inference(output_dir, custom_env_config, setup_dict, num_eps = 1)

    # CHOOSE ONE
    #evaluate(custom_env_config) # this one gives some wrong actions for the same observations
    #previous_code(custom_env_config) # this one gives the right actions


    # response_lst = cls.play_env()  # Environment creates the observations

     #  eval_cls.excel_n_plot(responses_by_distance, accepted_coalitions_by_distance )
