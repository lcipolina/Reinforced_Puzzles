''' Trying to see if I can simplify the script from RLLIB's example to have only one class AND MASKS'''

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from gymnasium.spaces import Discrete, Tuple
from gymnasium import spaces



class TorchAutoregressiveActionModel(TorchModelV2, nn.Module):
    """
    Combined model for autoregressive actions, integrating context encoding and action computation.
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

        # Number of discrete actions
        self.num_actions = action_space[0].n

        ''' ACA HAY UN ERROR'''
        # Shared context layers - produce a feature layer encoding the observations
        self.context_layer = SlimFC(
            in_size= 32, # This doesn't work as obs come in batches: obs_space.shape[0],  # Ensure this matches the observation space size
            out_size=2, # This gives nonsense: num_outputs,
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

        # Dynamically determine the hidden layer size based on the number of actions
        hidden_layer_size = num_outputs // 2

        # Layer to compute logits for action a1 based on the context input
        self.a1_logits = SlimFC(
            in_size=num_outputs,
            out_size=self.num_actions,
            activation_fn=None,
            initializer=normc_init_torch(0.01),
        )

        # Hidden layer for action a2
        self.a2_hidden = SlimFC(
            in_size=1,
            out_size=hidden_layer_size,
            activation_fn=nn.ReLU,
            initializer=normc_init_torch(1.0),
        )

        # Layer to compute logits for action a2 based on the hidden representation of a1
        self.a2_logits = SlimFC(
            in_size=hidden_layer_size,
            out_size=self.num_actions,
            activation_fn=None,
            initializer=normc_init_torch(0.01),
        )

        self._context = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass to encode observations into context vector, compute logits for actions, and apply masks.

        Args:
            input_dict (dict): Input dictionary containing observations and masks.
            state (Tensor): RNN state.
            seq_lens (Tensor): Sequence lengths for RNNs.

        Returns:
            Tuple[Tensor, Tensor]: Logits for actions a1 and a2, state.
        """
        obs = input_dict["obs"]

        self._context = self.context_layer(obs)

        # Extract masks from observations if present, otherwise default to None
        '''
        self.masks = {
            "mask_a1": input_dict["obs"].get("mask_a1", None),
            "mask_a2": input_dict["obs"].get("mask_a2", None)
        }
        '''
        self.masks = {
            "mask_a1": None,
            "mask_a2": None
        }

        # Compute logits for the first action based on the context input
        a1_logits = self.a1_logits(self._context)

        '''
        # Apply mask to a1 logits if provided
        if self.masks["mask_a1"] is not None:
            mask = self.masks["mask_a1"].float()
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)
            a1_logits += inf_mask
        '''

        # For autoregressive part: Compute hidden representation for the second action based on the first action input
        a1_input = torch.unsqueeze(a1_logits.argmax(dim=-1).float(), 1)  # Sample or take argmax from a1_logits

        a2_hidden_out = self.a2_hidden(a1_input)

        # Compute logits for the second action based on the hidden representation
        a2_logits = self.a2_logits(a2_hidden_out)

        '''
        # Apply mask to a2 logits if provided
        if self.masks["mask_a2"] is not None:
            mask = self.masks["mask_a2"].float()
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)
            a2_logits += inf_mask
        '''

        # Combine a1_logits and a2_logits into a single tensor
        combined_logits = torch.cat((a1_logits, a2_logits), dim=-1)

        return combined_logits, state

    def value_function(self):
        return torch.reshape(self.value_branch(self._context), [-1])
