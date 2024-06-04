import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from gymnasium.spaces import Discrete, Tuple



class ActionModel(nn.Module):
    def __init__(self, num_outputs, num_actions):
        super(ActionModel, self).__init__()

        # Dynamically determine the hidden layer size based on the number of actions
        hidden_layer_size = num_outputs // 2  # Heuristic: The hidden layer size is half the combined input size

        #  P(a1|obs) - First action output layer (i.e. no activation - later they will be passed through a softmax layer in the policy)
        self.a1_logits = SlimFC(
                    in_size=num_outputs,
                    out_size=num_actions,
                    activation_fn=None,
                    initializer=normc_init_torch(0.01),
                )

        # Second action hidden layer with ReLU activation
        self.a2_hidden = SlimFC(
            in_size=1,
            out_size=hidden_layer_size,
            activation_fn=nn.ReLU,
            initializer=normc_init_torch(1.0),
        )
        # P(a2 | a1)- Second action logits - no activation function to output raw logits (later they will be passed through a softmax layer in the policy)
        self.a2_logits = SlimFC(
            in_size=hidden_layer_size,
            out_size=num_actions,   # Output dimension based on the number of discrete actions
            activation_fn=None,
            initializer=normc_init_torch(0.01),
        )

    def forward(self, ctx_input, a1_input):
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

        # Instantiate action model - Called by the Distrib class and the FWD computes both a1 and a2 logits (lazy programming) - then the action model discards one
        self.action_module = ActionModel(num_outputs, self.num_actions)

        self._context = None

    def forward(self, input_dict, state, seq_lens):
        self._context = self.context_layer(input_dict["obs"])
        return self._context, state

    def value_function(self):
        # Compute and return the value function from the context vector
        return torch.reshape(self.value_branch(self._context), [-1])
