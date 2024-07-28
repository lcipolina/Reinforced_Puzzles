'''THIS WORKS - everything in one class PLUS MASK!!'''

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from gymnasium.spaces import Discrete, Tuple
from gymnasium import spaces


class CombinedAutoregressiveActionModel(TorchModelV2, nn.Module):
    """
    Combined model for autoregressive actions, integrating both context encoding and action logits computation.

    Args:
        obs_space (Space): Observation space.
        action_space (Space): Action space.
        num_outputs (int): Number of output features.
        model_config (dict): Model configuration.
        name (str): Model name.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if not (isinstance(action_space, Tuple) and all(isinstance(space, Discrete) for space in action_space)):
            raise ValueError("This model only supports a tuple of discrete action spaces")

        self.num_actions = action_space[0].n
        self.hidden_layer_size = num_outputs // 2

        # Context layer to encode observations
        self.context_layer = SlimFC(
            in_size=obs_space.shape[0],
            out_size=num_outputs,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh,
        )

        # Value function branch
        self.value_branch = SlimFC(
            in_size=num_outputs,
            out_size=1,
            initializer=normc_init_torch(0.01),
            activation_fn=None,
        )

        # Layer to compute logits for action a1
        self.a1_logits = SlimFC(
            in_size=num_outputs,
            out_size=self.num_actions,
            activation_fn=None,
            initializer=normc_init_torch(0.01),
        )

        # Hidden layer for action a2
        self.a2_hidden = SlimFC(
            in_size=1,
            out_size=self.hidden_layer_size,
            activation_fn=nn.ReLU,
            initializer=normc_init_torch(1.0),
        )

        # Layer to compute logits for action a2
        self.a2_logits = SlimFC(
            in_size=self.hidden_layer_size,
            out_size=self.num_actions,
            activation_fn=None,
            initializer=normc_init_ttorch(0.01),
        )

        self._context = None
        self.masks = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass to encode observations into context vector and prepare masks.

        Args:
            input_dict (dict): Input dictionary containing observations and masks.
            state (Tensor): RNN state.
            seq_lens (Tensor): Sequence lengths for RNNs.

        Returns:
            Tuple[Tensor, Tensor]: Encoded context vector and state.
        """
        obs = input_dict["obs"]["features"]  # Extract features from the observations
        self._context = self.context_layer(obs)

        # Extract masks from observations
        self.masks = {
            "mask_a1": input_dict["obs"].get("mask_a1", None),
            "mask_a2": input_dict["obs"].get("mask_a2", None)
        }

        return self._context, state

    def compute_action_logits(self, ctx_input, a1_input):
        """
        Compute logits for actions a1 and a2, applying masks if provided.

        Args:
            ctx_input (Tensor): Context vector from observations.
            a1_input (Tensor): Input from the first action.

        Returns:
            Tuple[Tensor, Tensor]: Logits for actions a1 and a2.
        """
        # Compute logits for the first action based on the context input
        a1_logits = self.a1_logits(ctx_input)

        # Apply mask to a1 logits if provided
        if self.masks["mask_a1"] is not None:
            mask = self.masks["mask_a1"].float()
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)
            a1_logits += inf_mask

        # Compute hidden representation for the second action based on the first action input
        a2_hidden_out = self.a2_hidden(a1_input)

        # Compute logits for the second action based on the hidden representation
        a2_logits = self.a2_logits(a2_hidden_out)

        # Apply mask to a2 logits if provided
        if self.masks["mask_a2"] is not None:
            mask = self.masks["mask_a2"].float()
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)
            a2_logits += inf_mask

        return a1_logits, a2_logits

    def value_function(self):
        """
        Compute and return the value function from the context vector.

        Returns:
            Tensor: Value function output.
        """
        return torch.reshape(self.value_branch(self._context), [-1])
