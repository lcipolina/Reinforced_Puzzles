''' Train with Ray Air and Custom Model and Action Masking'''


import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
#from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

# TODO: Action Masking settings:
#https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py


from env import PuzzleGymEnv
from policy import MaskedActionModel as CustomTorchModel

# Register the custom model
ModelCatalog.register_custom_model("masked_action_model", CustomTorchModel)

CPU_NUM = 7
env_config_dict = {
        'sides': [5, 6, 7, 8],  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
        'num_pieces': 4,
        }

def setup():
    trainer_config = (PPOConfig()
                      .environment(env=PuzzleGymEnv,
                                   env_config= env_config_dict,
                                   )
                      .training(train_batch_size=1000,
                                sgd_minibatch_size=64,  # These are the number of samples that are collected before a gradient step is taken.
                                 entropy_coeff=0.2,
                                  kl_coeff=0.1,
                                 model = {"custom_model":  "masked_action_model",
                                            "_disable_preprocessor_api": False,  # if True, dicts are converted to Tensors - and we can't distinguish between different observations and masks
                                          } )
                      .rollouts(num_rollout_workers=CPU_NUM, num_envs_per_worker=1, rollout_fragment_length='auto')
                      .framework("torch")
                     # .debugging(seed=42)
                     # We need to disable preprocessing of observations, because preprocessing
                    # would flatten the observation dict of the environment.
                    .experimental(
                        _disable_preprocessor_api=False, # Do not flatten the observation dict of the environment
                    )

                      )      #   .rl_module(_enable_rl_module_api=False) #to keep using the old Mod



    # Setup the tuner and training
    tuner = tune.Tuner("PPO", param_space = trainer_config,
                              run_config=air.RunConfig(
                                name = 'puzzle' , #self.experiment_name,
                                stop={"training_iteration": 30},
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10,
                                                                        checkpoint_at_end=False,
                                                                        num_to_keep=3),
                                local_dir="ray_results",
                                verbose=2,
                       ))

    result_grid = tuner.fit() #train the model
    # Get reward per policy
    best_result_grid = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
    return best_result_grid

def main():

    storage_address = "/Users/lucia/Desktop/Art_Project/000-A_Pompei/Repair_project/CODE_puzzle/ray_results/puzzle" #New requirement
    ray.init(ignore_reinit_error=True,
             local_mode=True,
             storage = storage_address)
    results = setup()
    ray.shutdown()

    #best_checkpoint = results.get_best_checkpoint(metric="episode_reward_mean", mode="max")
    #print("Best checkpoint path:", best_checkpoint)

if __name__ == "__main__":
    main()
