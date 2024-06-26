import os
import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
from ray.tune.logger import pretty_print

'''
Actions: (a1, a2) both can take 2 values: 0 or 1.
Action 'a2' depends on the value of action a1.

'''


# Action distribution for autoregressive actions.
#from ray.rllib.examples._old_api_stack.models.autoregressive_action_dist import TorchBinaryAutoregressiveDistribution
from autoreg_action_dist import TorchAutoregressiveCategoricalDistribution as TorchBinaryAutoregressiveDistribution

# Network model for autoregressive actions.
#from ray.rllib.examples._old_api_stack.models.autoregressive_action_model import TorchAutoregressiveActionModel
from autoreg_model import TorchAutoregressiveActionModel


# Environment for this example.
#from ray.rllib.examples.envs.classes.correlated_actions_env import CorrelatedActionsEnv
from env import CorrelatedActionsEnv

if __name__ == "__main__":
    # Default values
    run_algorithm = "PPO"
    framework = "torch"
    num_cpus = 0
    as_test = False
    stop_iters = 200
    stop_timesteps = 100000
    stop_reward = 200.0
    no_tune = False  # If True, run manual training loop without Tune
    local_mode = True
    no_autoreg = False

    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    # Register and configure autoregressive action model and dist
    ModelCatalog.register_custom_model(
        "autoregressive_model", TorchAutoregressiveActionModel
    )
    ModelCatalog.register_custom_action_dist(
        "binary_autoreg_dist", TorchBinaryAutoregressiveDistribution
    )

    # Generic config
    config = (
        get_trainable_cls(run_algorithm)
        .get_default_config()
        .experimental(_enable_new_api_stack=False)
        .environment(CorrelatedActionsEnv)
        .framework(framework)
        .training(gamma=0.5)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # Use registered model and dist in config if not disabled
    if not no_autoreg:
        config.model.update(
            {
                "custom_model": "autoregressive_model",
                "custom_action_dist": "binary_autoreg_dist",
            }
        )

    # Stop conditions
    stop = {
        "training_iteration": stop_iters,
        "num_env_steps_sampled_lifetime": stop_timesteps,
        "env_runner_results/episode_return_mean": stop_reward,
    }

    # Manual training loop without Tune
    if no_tune:
        if run_algorithm != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        config.algo_class = run_algorithm
        algo = config.build()
        for _ in range(stop_iters):
            result = algo.train()
            print(pretty_print(result))
            if (
                result["timesteps_total"] >= stop_timesteps
                or result["episode_reward_mean"] >= stop_reward
            ):
                break

        # Manual test loop
        print("Finished training. Running manual test/inference loop.")
        env = CorrelatedActionsEnv(_)
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            a1, a2 = algo.compute_single_action(obs)
            next_obs, reward, done, truncated, _ = env.step((a1, a2))
            print(f"Obs: {obs}, Action: a1={a1} a2={a2}, Reward: {reward}")
            obs = next_obs
            total_reward += reward
        print(f"Total reward in test episode: {total_reward}")
        algo.stop()

    # Run with Tune for automatic environment and algorithm creation
    else:
        tuner = tune.Tuner(
            run_algorithm, run_config=air.RunConfig(stop=stop, verbose=2), param_space=config
        )
        results = tuner.fit()

        if as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, stop_reward)

    ray.shutdown()