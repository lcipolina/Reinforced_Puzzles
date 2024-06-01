''' Train Single Agent PPO on Ray 2.12'''

from ray.rllib.algorithms.ppo import PPOConfig


def get_sarl_trainer_config(env_generator,
                            custom_env_config,
                            _train_batch_size,
                              lr_start, lr_time,
                              lr_end, num_cpus,
                              _seed = 42,
                           ):

    trainer_config = (PPOConfig()
                            .environment(env=env_generator,
                                    env_config= custom_env_config,
                                    )
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