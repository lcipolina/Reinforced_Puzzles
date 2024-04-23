''' Train with Ray Air and Custom Model and Action Masking'''


import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()


from env import PuzzleGymEnv


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.fc = nn.Linear(obs_space.shape[0], num_outputs)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        state = obs["state"]
        action_mask = obs["action_mask"]

        logits = self.fc(state)
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e15)
        masked_logits = logits + inf_mask
        return masked_logits, state


# Register the custom model
ModelCatalog.register_custom_model("my_custom_model", CustomTorchModel)

CPU_NUM = 7

#TODO: check if the model needs to be registered before the trainer config
#TODO: check if others also implement the mask in the OBSERVATION space!!

def setup():
    trainer_config = (PPOConfig()
                      .environment(env=PuzzleGymEnv,
                                   env_config={'sides': [1, 2, 3, 4]},
                                   disable_env_checking=True)
                      .training(train_batch_size=1024,
                                 model = {"custom_model": "my_custom_model",
                                            "_disable_preprocessor_api": True,
                                          } ,
                                  _enable_rl_module_api=False )
                      .rollouts(num_rollout_workers=CPU_NUM, num_envs_per_worker=1, rollout_fragment_length='auto')
                      .framework("torch")
                      .debugging(seed=42)
                     .experimental(_disable_preprocessor_api=True)
                      )      #   .rl_module(_enable_rl_module_api=False) #to keep using the old Mod

#TODO: test, understad and document this, regarding the settings.
# https://stackoverflow.com/questions/77425959/function-parameters-in-customizing-models-in-ray-rllib

    # Setup the tuner and training
    tuner = tune.Tuner("PPO", param_space = trainer_config,
                           run_config=air.RunConfig(
                                name =  'puzzle' , #self.experiment_name,
                                stop={"training_iteration": 5},
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10,
                                                                        checkpoint_at_end=False,
                                                                        num_to_keep=5),
                                local_dir="ray_results",
                                verbose=3,
                       ))

    result_grid = tuner.fit() #train the model
    # Get reward per policy
    best_result_grid = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
    return best_result_grid

def main():
    ray.init(ignore_reinit_error=True)
    results = setup()
    ray.shutdown()

    #best_checkpoint = results.get_best_checkpoint(metric="episode_reward_mean", mode="max")
    #print("Best checkpoint path:", best_checkpoint)

if __name__ == "__main__":
    main()
