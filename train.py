import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from env import PuzzleGymEnv
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()




def setup():
    trainer_config = (PPOConfig()
                      .environment(env=PuzzleGymEnv,
                                   env_config={'sides': [1, 2, 3, 4]},
                                   disable_env_checking=True)
                      .training(train_batch_size=1024)
                      .rollouts(num_rollout_workers=7, num_envs_per_worker=1, rollout_fragment_length='auto')
                      .framework("torch")
                      .debugging(seed=42)
                      ) #                     .rl_module(_enable_rl_module_api=False) #to keep using the old Mod

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
