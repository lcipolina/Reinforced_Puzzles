import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from gymnasium.spaces import Discrete, Tuple

'''
THIS SCRIPT IS CLOSER TO THE ORIGINAL EXAMPLE FROM RLLIB:
https://github.com/ray-project/ray/blob/master/rllib/examples/autoregressive_action_dist.py

Code Flow:
1. Environment: Provides observations.
   ↓
2. Observation: Passed to TorchAutoregressiveActionModel.
   ↓
3. TorchAutoregressiveActionModel: Encodes observations into a context vector.
   ↓
4. Context Vector: Used by ActionModel to compute logits.
   ↓
5. ActionModel: Computes logits for actions a1 and a2 based on the context vector and previous actions.
   ↓
6. TorchAutoregressiveCategoricalDistribution: Uses the logits to sample actions a1 and a2|a1.


Classes:
- TorchAutoregressiveCategoricalDistribution: Handles autoregressive action sampling and log probability calculations.
- ActionModel: Computes logits for the autoregressive actions.
- TorchAutoregressiveActionModel: Integrates ActionModel into the RLlib framework, providing context encoding and value function computation.
'''

class ActionModel(nn.Module):
    """
    Neural network model to compute the logits for actions a1 and a2.

    Args:
        num_outputs (int): Number of output features from the context layer.
        num_actions (int): Number of discrete actions.
    """


    def __init__(self, num_outputs, num_actions):
        super(ActionModel, self).__init__()

        # Dynamically determine the hidden layer size based on the number of actions
        hidden_layer_size = num_outputs // 2  # TODO: Heuristic: The hidden layer size is half the combined input size

        # Layer to compute logits for action a1 based on the context input
        #  P(a1|obs) - First action output layer (i.e. no activation - later they will be passed through a softmax layer in the policy)
        self.a1_logits = SlimFC(
                    in_size=num_outputs,
                    out_size=num_actions,
                    activation_fn=None,
                    initializer=normc_init_torch(0.01),
                )

        # Hidden layer for action a2
        # Second action hidden layer with ReLU activation
        a1_input_size = 1            # Input size is the number of a1 actions  #TODO: review this!
        self.a2_hidden = SlimFC(
            in_size=a1_input_size,   # Input size is 1 because the input is the action taken in the previous step
            out_size=hidden_layer_size,
            activation_fn=nn.ReLU,
            initializer=normc_init_torch(1.0),
        )

        # Layer to compute logits for action a2 based on the hidden representation of a1
        # P(a2 | a1)- Second action logits - no activation function to output raw logits (later they will be passed through a softmax layer in the policy)
        self.a2_logits = SlimFC(
            in_size=hidden_layer_size,
            out_size=num_actions,   # Output dimension based on the number of discrete actions
            activation_fn=None,
            initializer=normc_init_torch(0.01),
        )

    def forward(self, ctx_input, a1_input):
        """
        Forward pass to compute logits for actions a1 and a2.
        Args:
            ctx_input (Tensor): Context vector from observations.
            a1_input (Tensor): Input from the first action.
        Returns:
            Tuple[Tensor, Tensor]: Logits for actions a1 and a2.
        """
        # Compute logits for both actions based on the combined input
        # The combined input is the context vector and the logits for the first action (a1)
        # Input is as required by the Action Distribution (context vector, logits for the first action)

        # Compute logits for the first action based on the context input
        a1_logits = self.a1_logits(ctx_input)

        # Compute hidden representation for the second action based on the first action input
        a2_hidden_out = self.a2_hidden(a1_input)

        # Compute logits for the second action based on the hidden representation
        a2_logits = self.a2_logits(a2_hidden_out)
        return a1_logits, a2_logits



class TorchAutoregressiveActionModel(TorchModelV2, nn.Module):
    """
    Model class integrating ActionModel into the RLlib framework.
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

        # Shared context layers - produce a feature layer encoding the observations
        self.context_layer = SlimFC(  # SlimFC is a helper class that applies a linear layer and initialization
            in_size=obs_space.shape[0],
            out_size=num_outputs,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh,
        )

        # Value function branch - no activation function to allow any real number output
        self.value_branch = SlimFC(
            in_size=num_outputs,
            out_size=1,
            initializer=normc_init_torch(0.01),
            activation_fn=None,
        )

        # Action model instance to compute action logits
        # Instantiate action model - Called by the Distrib class and the FWD computes both a1 and a2 logits (lazy programming) - then the action model discards one
        self.action_module = ActionModel(num_outputs, self.num_actions)

        self._context = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass to encode observations into context vector.
        Args:
            input_dict (dict): Input dictionary containing observations.
            state (Tensor): RNN state.
            seq_lens (Tensor): Sequence lengths for RNNs.
        Returns:
            Tuple[Tensor, Tensor]: Encoded context vector and state.
        """
        self._context = self.context_layer(input_dict["obs"])
        return self._context, state

    def value_function(self):
        # Compute and return the value function from the context vector
        return torch.reshape(self.value_branch(self._context), [-1])
