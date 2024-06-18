''' Train Single Agent PPO on Ray 2.12

    Uses a custom Model with action masking
'''

from ray.rllib.algorithms.ppo import PPOConfig
from typing import Dict                                               # for callbacks
from ray.rllib.evaluation import RolloutWorker, Episode               # for callbacks
from ray.rllib.policy import Policy                                   #for callbacks
from ray.rllib.env import BaseEnv                                     # for callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks           # for callbacks



#------------------------------------------------------------------------------------------------
# Single-agent RL
#------------------------------------------------------------------------------------------------
def get_sarl_trainer_config(env_class,
                            custom_env_config,
                            setup_dict,
                            lr_start = None,
                            lr_time = None,
                            lr_end = None
                           ):
    '''Returns the configuration for the PPO trainer
       Args:
              env_class: The environment class (NOT the generator! as it registers the env)
       OBS: Custom model and action distribution have been registered in their defn scripts.
    '''

    _train_batch_size  = setup_dict['train_batch_size']
    num_cpus           = setup_dict['cpu_nodes']
    _seed              = setup_dict.get('seed', 42)

    trainer_config = (PPOConfig()
                            .environment(env=env_class, # if we pass it like this, we don't need to register the env
                                         env_config= custom_env_config)
                            .training(train_batch_size=_train_batch_size,  # Number of samples that are collected before a gradient step is taken.
                                    sgd_minibatch_size=64,  # These are the number of samples that are used for each SGD iteration.
                                # entropy_coeff=0.2,  #it doesnt seem to change much
                                # kl_coeff=0.01,
                                    model = {"custom_model":  "masked_action_model",
                                             "_disable_preprocessor_api": False,  # if True, dicts are converted to Tensors - and we can't distinguish between different observations and masks
                                            } )
                            .rollouts(num_rollout_workers=num_cpus, num_envs_per_env_runner=1, rollout_fragment_length='auto')
                            .framework("torch")
                             .debugging(seed=_seed)
                        # We need to disable preprocessing of observations, because preprocessing
                        # would flatten the observation dict of the environment - and we need it for the action mask
                        .experimental(
                            _disable_preprocessor_api= False, # Do no flatten the observation dict of the environment
                        )

                        )      #   .rl_module(_enable_rl_module_api=False) #to keep using the old Mod


    return trainer_config


#------------------------------------------------------------------------------------------------
# Hierarchical RL
#------------------------------------------------------------------------------------------------
#_____________________________________________________________________________
# Custom callbacks
# Get reward per agent (not provided in RLLIB)
# WandB Callbacks - Just logs results and metrics on each iteration

class On_step_callback(DefaultCallbacks):
    '''To get rewards per agent
       Needs to be run with default 'verbose = 3' value to be displayed on screen
       #https://github.com/ray-project/ray/blob/master/rllib/evaluation/metrics.py#L229-L231
    '''

# TODO: I am not sure if this is what we need. Look at my script here:
# /Users/lucia/Desktop/LuciaArchive/000_A_MY_RESEARCH/00-My_Papers/Ridesharing/000-A-RidesharingMARL/00-Codes/coalitions/A-coalitions_paper/C_ppo_config.py

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                  policies: Dict[str, Policy], episode: Episode,**kwargs):
        '''Calculates the reward per agent at the STEP of the episode. Displays on Tensorboard and on the console   '''
        #The advantage of these ones is that it calculates the Max-Mean-Min and it prints on TB
        #NOTE: custom_metrics only take scalars
        my_dict = {}  #Needs this as metric name needs to be a string
        for key, values in episode.agent_rewards.items():
            my_dict[str(key)] = values
            #my_dict[str(key)+'_std_dev'] = np.std(values) #this one comes empty
            episode.custom_metrics.update(my_dict)
#_____________________________________________________________________________
# POLICY MAPPING EXTRAS- For Multi-Policy Training
# 1) Define the policies definition dict: done in the trainer_config

# 2) Maps agents-> to the defined policies
# The mapping here is M (agents) -> N (policies), where M >= N.
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''Maps agents to the defined policies. Ex: agent_id = "low_level_1" -> "low_level_policy"'''
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"
#_____________________________________________________________________________
def get_env_spaces(env_class, custom_env_config):
    # Initialize the environment with the required dictionary
    env =  env_class(custom_env_config)
    high_obs_space = env.observation_space["high_level_agent"]
    high_action_space = env.action_space["high_level_agent"]
    low_obs_space = env.observation_space["low_level_agent"]
    low_action_space = env.action_space["low_level_agent"]
    env.close()  # Ensure you clean up the environment instance
    return high_obs_space, high_action_space, low_obs_space, low_action_space

def get_marl_hrl_trainer_config(env_class,
                            custom_env_config,
                            setup_dict,
                            lr_start = None,
                            lr_time = None,
                            lr_end = None
                           ):

    high_obs_space, high_action_space, low_obs_space, low_action_space = get_env_spaces(env_class, custom_env_config)

    _train_batch_size  = setup_dict['train_batch_size']
    num_cpus           = setup_dict['cpu_nodes']
    _seed              = setup_dict.get('seed', 42)

    trainer_config = (
            PPOConfig()
            .environment(env=env_class, # if we pass it like this, we don't need to register the env
                        env_config= custom_env_config)
            .training(train_batch_size=_train_batch_size,  # Number of samples that are collected before a gradient step is taken.
                      sgd_minibatch_size=64,  # These are the number of samples that are used for each SGD iteration.
                       # entropy_coeff=0.2,  #it doesnt seem to change much
                       # kl_coeff=0.01,
                      model = {"_disable_preprocessor_api": False,  # if True, dicts are converted to Tensors - and we can't distinguish between different observations and masks
                                            } )
            .rollouts(num_rollout_workers=num_cpus, num_envs_per_env_runner=1, rollout_fragment_length='auto')
            .framework("torch")
            .debugging(seed=_seed)
                        # We need to disable preprocessing of observations, because preprocessing
                        # would flatten the observation dict of the environment - and we need it for the action mask
            .experimental(
                  _disable_preprocessor_api= False, # Do no flatten the observation dict of the environment
                        )
            .multi_agent(  #https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/two_algorithms.py
                policies= {
               "high_level_policy": (None, high_obs_space, high_action_space, {"model": {
                    "custom_model": "custom_model_high"
                   # "custom_action_dist": CustomActionDistributionHigh,
                }}),
            "low_level_policy": (None, low_obs_space, low_action_space,{"model": {
                    "custom_model": "custom_model_low",
                   # "custom_action_dist": CustomActionDistributionLow,
               }}),
              },
                policy_mapping_fn=policy_mapping_fn,
            ) # end of multi_agent
            .callbacks(On_step_callback)
        ) # end of trainer_config
    return trainer_config