''' Hierarchical RL Example'''



import ray, os
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.classes.windy_maze_env import (
     WindyMazeEnv,
#     HierarchicalWindyMazeEnv,
)


# from ray.rllib.utils.metrics import (
#     ENV_RUNNER_RESULTS,
#     EPISODE_RETURN_MEAN,
#     NUM_ENV_STEPS_SAMPLED_LIFETIME,
# )
from ray.rllib.utils.test_utils import check_learning_achieved



import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Dict
from gymnasium.spaces import Box, Discrete, Tuple
from ray.rllib.env import MultiAgentEnv

import logging



logger = logging.getLogger(__name__)


class HierarchicalWindyMazeEnv(MultiAgentEnv):
    '''Hierarchical Windy Maze Environment'''
    def __init__(self, env_config):
        super().__init__()
        self.flat_env = WindyMazeEnv(env_config)

    def reset(self, *, seed=None, options=None):
        self.cur_obs, infos = self.flat_env.reset()
        self.current_goal = None
        self.steps_remaining_at_level = None
        self.num_high_level_steps = 0
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        return {
            "high_level_agent": self.cur_obs,
        }, {"high_level_agent": infos}

    def step(self, action_dict):

        # The action dict tells us who has taken the action
        # Need to process the action and setup the Obs for the next agent
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action):
        logger.debug("High level agent sets goal")
        self.current_goal = action
        self.steps_remaining_at_level = 25
        self.num_high_level_steps += 1
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
        rew = {self.low_level_agent_id: 0}
        done = truncated = {"__all__": False}
        return obs, rew, done, truncated, {}

    def _low_level_step(self, action):
        logger.debug("Low level agent step {}".format(action))
        self.steps_remaining_at_level -= 1
        cur_pos = tuple(self.cur_obs[0])
        goal_pos = self.flat_env._get_new_pos(cur_pos, self.current_goal)

        # Step in the actual env
        f_obs, f_rew, f_terminated, f_truncated, info = self.flat_env.step(action)
        new_pos = tuple(f_obs[0])
        self.cur_obs = f_obs

        # Calculate low-level agent observation and reward
        obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
        if new_pos != cur_pos:
            if new_pos == goal_pos:
                rew = {self.low_level_agent_id: 1}
            else:
                rew = {self.low_level_agent_id: -1}
        else:
            rew = {self.low_level_agent_id: 0}

        # Handle env termination & transitions back to higher level.
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        if f_terminated or f_truncated:
            terminated["__all__"] = f_terminated
            truncated["__all__"] = f_truncated
            logger.debug("high level final reward {}".format(f_rew))
            rew["high_level_agent"] = f_rew
            obs["high_level_agent"] = f_obs
        elif self.steps_remaining_at_level == 0:
            terminated[self.low_level_agent_id] = True
            truncated[self.low_level_agent_id] = False
            rew["high_level_agent"] = 0
            obs["high_level_agent"] = f_obs

        return obs, rew, terminated, truncated, {self.low_level_agent_id: info}


if __name__ == "__main__":

    #******************************************************************************
    # Part 1: Train RLlib Hierarchical Windy Maze Environment
    # *****************************************************************************
    # Default values
    use_flat = False
    framework = "torch"
    as_test = False
    stop_iters = 200
    stop_timesteps = 100000
    stop_reward = 0.0
    local_mode = False

    ray.init(local_mode=local_mode)

    stop = {
        'training_iteration': 3,
        #'episodes_timesteps_total': stop_timesteps,
       # stop_reward: stop_reward,
    }

    if use_flat:
        results = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(stop=stop),
            param_space=(
                PPOConfig()
                .environment(WindyMazeEnv)
                .env_runners(num_env_runners=0)
                .framework(framework)
            ).to_dict(),
        ).fit()
    else:
        maze = WindyMazeEnv(None)

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("low_level_"):
                return "low_level_policy"
            else:
                return "high_level_policy"

        config = (
            PPOConfig()
            .environment(HierarchicalWindyMazeEnv)
            .framework(framework)
            .env_runners(num_env_runners=0)
            .training(entropy_coeff=0.01)
            .multi_agent(
                policies={
                    "high_level_policy": (
                        None,
                        maze.observation_space,
                        Discrete(4),
                        PPOConfig.overrides(gamma=0.9),
                    ),
                    "low_level_policy": (
                        None,
                        Tuple([maze.observation_space, Discrete(4)]),
                        maze.action_space,
                        PPOConfig.overrides(gamma=0.0),
                    ),
                },
                policy_mapping_fn=policy_mapping_fn,
            )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )

        results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=1),
        ).fit()

    if as_test:
        check_learning_achieved(results, stop_reward)

    ray.shutdown()
