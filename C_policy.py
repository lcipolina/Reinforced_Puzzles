import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from gymnasium.spaces import Discrete, Tuple, Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch


''' Defines custom Policies with action masking.
    Compute raw logits for each part of the action space and apply masks to the logits.
    RLLIB will then sample actions from the action space based on the logits.
    Registers the custom models with RLLIB.
'''

############################################################################################################################
# Single-agent RL with masking
############################################################################################################################
class CustomMaskedModel(TorchModelV2, nn.Module):
    '''Model used for single-agent training with action masking.'''

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        #_____________________________________________________________________________________________________________________________________________
        # Define the network architecture: i.e. the order of operations in the forward pass and their sizes
        # - Shared layers and separate output layers for each part of the action space
        # Input dimensions are: [batch_size, observation_size]
        # Output dimensions are: [batch_size, number_of_actions] -  categorical logits for each part of the action space  (or mu, sigma in cts space)
        #_____________________________________________________________________________________________________________________________________________

        # Define the shared network with custom hidden layer sizes
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),   # ONLY POSITIVE VALUES!
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Define separate output layers for each categorical dimension of the action space
        # Each output layer will produce logits corresponding to the number of possible actions for that dimension.
        # RLLIB then uses these logits to sample actions from the action space. RLLIB then softmaxes the logits to get probabilities.
        self.logits_active_piece_id = nn.Linear(128, action_space.nvec[0])             # action_space.nvec[0] is the number of pieces - Select the active piece
        # This one will not have a mask, as all sides are available for the active piece
        self.logits_active_piece_side_index = nn.Linear(128, action_space.nvec[1])     # action_space.nvec[1] is the number of target pieces - Select side of the active piece
        # Target side depends on the target piece, so the number of possible actions is the number of target pieces times the number of sides
        self.logits_target_piece_and_side = nn.Linear(128, action_space.nvec[2])       # Combined dimensions: [num_pieces, num_sides]  - Select the target piece and side of the target piece

        # Define the value head  -  Value function network
        self.value_head = nn.Linear(128, 1)


    def forward(self, input_dict, state, seq_lens):
        #___________________________________________________________________________
        # Do the actual forward pass over the architecture defined above
        # Output are masked logits for each part of the action space
        #___________________________________________________________________________

        self.shared_layers_output = self.shared_layers(input_dict["obs_flat"])                           # Pass through the shared layers. Dimensions: [batch_size (32), 128].

        # 1. Compute logits for each part of the action space
        logits_active_piece_id         = self.logits_active_piece_id(self.shared_layers_output)          # shape: [batch_size (32), num_pieces]
        logits_active_piece_side_index = self.logits_active_piece_side_index(self.shared_layers_output)  # shape: [batch_size(32), num_target_pieces]
        logits_target_piece_and_side   = self.logits_target_piece_and_side(self.shared_layers_output)    # shape: [batch_size(32), num_pieces * num_sides]

        # 2. Apply masks to the logits to prevent invalid actions
        # Only Available Pieces (i.e. not placed) can be selected as current piece
        if "mask_piece_id" in input_dict["obs"]:
            mask = input_dict["obs"]["mask_piece_id"].float()  # mask are defined the other way around from what we need them :/
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)  # Calculate the inf_mask where mask values are 0 (invalid) and will have -inf after log transformation
            logits_active_piece_id += inf_mask

        # Only Available Target (placed) Pieces Can be Selected as Target Piece - if the piece is placed and has at least one available side
        if "mask_target_side_index" in input_dict["obs"]:
            mask = input_dict["obs"]["mask_target_side_index"].float()
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)
            logits_target_piece_and_side += inf_mask

        # Concatenate all logits into a single tensor as required by RLLIB
        all_logits = torch.cat([
            logits_active_piece_id,         # selected_piece_index
            logits_active_piece_side_index, # selected_side_index
            logits_target_piece_and_side    # target_piece_index, target_side_index
        ], dim=1)

        return all_logits, state

    def value_function(self):
        value = self.value_head(self.shared_layers_output)
        return value.squeeze(-1)  # Remove any unnecessary dimensions to match [batch_size]


####################################################################################################################################
# Policies for multi-agent Hierarchical RL - High and Low Level Agents. The difference between the policies is mostly on the masking
####################################################################################################################################
class CustomModelHigh(TorchModelV2, nn.Module):
    """Custom model for the high-level agent."""
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: Dict[str, Any], name: str) -> None:
        super(CustomModelHigh, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        #____________________________________________________________________________________________________________________________________________
        # Define the network architecture: i.e. the order of operations in the forward pass and their sizes
        # Define shared layers and separate output layers for each part of the action space
        # Input dimensions are: [batch_size, observation_size]
        # Output dimensions are: [batch_size, number_of_actions]] -  categorical logits for each part of the action space  (or mu, sigma in cts space)
        #____________________________________________________________________________________________________________________________________________

        print("entering CustomModelHigh")

        # Define the shared network with custom hidden layer sizes
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),                              # ONLY POSITIVE VALUES!
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Define separate output layers for each categorical dimension of the action space
        # Each output layer will produce logits corresponding to the number of possible actions for that dimension.
        # RLLIB then uses these logits to sample actions from the action space. RLLIB then softmaxes the logits to get probabilities.

        # Target side depends on the target piece, so the number of possible actions is the number of target pieces times the number of sides
        self.logits_target_piece_and_side = nn.Linear(128, action_space.n)       # Combined dimensions: [num_pieces, num_sides]  - Select the target piece and side of the target piece

        # Define the value head  -  Value function network
        self.value_head = nn.Linear(128, 1)


    def forward(self, input_dict: Dict[str, torch.Tensor], state: List[torch.Tensor], seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        #___________________________________________________________________________
        # Do the actual forward pass over the architecture defined above
        # Output are masked logits for each part of the action space
        #___________________________________________________________________________

        self.shared_layers_output = self.shared_layers(input_dict["obs_flat"])                           # Pass through the shared layers. Dimensions: [batch_size (32), 128].

        # 1. Compute logits for each part of the action space
        logits_target_piece_and_side   = self.logits_target_piece_and_side(self.shared_layers_output)    # shape: [batch_size(32), num_pieces * num_sides]

        # 2. Apply masks to the logits to prevent invalid actions
        # Only Available Target (placed) Pieces Can be Selected as Target Piece - if the piece is placed and has at least one available side
        if "mask_target_side_index" in input_dict["obs"]:
            mask = input_dict["obs"]["mask_target_side_index"].float()
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)
            logits_target_piece_and_side += inf_mask

        # Concatenate all logits into a single tensor as required by RLLIB
        all_logits =logits_target_piece_and_side    # target_piece_index, target_side_index
        return all_logits, state

    def value_function(self) -> torch.Tensor:
        value = self.value_head(self.shared_layers_output)
        return value.squeeze(-1)  # Remove any unnecessary dimensions to match [batch_size]

class CustomModelLow(TorchModelV2, nn.Module):
    """Custom model for the low-level agent."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModelLow, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        print("entering CustomModelLow")

        #____________________________________________________________________________________________________________________
        # Define the network architecture: i.e. the order of operations in the forward pass and their sizes
        # - Shared layers and separate output layers for each part of the action space
        # Input dimensions are: [batch_size, observation_size]
        # Output dimensions are: [batch_size, number_of_actions]] -  categorical logits for each part of the action space  (or mu, sigma in cts space)
        #____________________________________________________________________________________________________________________


        # Define the shared network with custom hidden layer sizes
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),   # ONLY POSITIVE VALUES!
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Define separate output layers for each categorical dimension of the action space
        # Each output layer will produce logits corresponding to the number of possible actions for that dimension.
        # RLLIB then uses these logits to sample actions from the action space. RLLIB then softmaxes the logits to get probabilities.
        self.logits_active_piece_id = nn.Linear(128, action_space.nvec[0])             # action_space.nvec[0] is the number of pieces - Select the active piece
        # This one will not have a mask, as all sides are available for the active piece
        self.logits_active_piece_side_index = nn.Linear(128, action_space.nvec[1])     # action_space.nvec[1] is the number of target pieces - Select side of the active piece

        # Define the value head  -  Value function network
        self.value_head = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):
         #___________________________________________________________________________
        # Do the actual forward pass over the architecture defined above
        # Output are masked logits for each part of the action space
        #___________________________________________________________________________

        self.shared_layers_output = self.shared_layers(input_dict["obs_flat"])                           # Pass through the shared layers. Dimensions: [batch_size (32), 128].

        # 1. Compute logits for each part of the action space
        logits_active_piece_id         = self.logits_active_piece_id(self.shared_layers_output)          # shape: [batch_size (32), num_pieces]
        logits_active_piece_side_index = self.logits_active_piece_side_index(self.shared_layers_output)  # shape: [batch_size(32), num_target_pieces]

        # 2. Apply masks to the logits to prevent invalid actions
        # Only Available Pieces (i.e. not placed) can be selected as current piece
        if "mask_piece_id" in input_dict["obs"]:
            mask = input_dict["obs"]["mask_piece_id"].float()  # mask are defined the other way around from what we need them :/
            inf_mask = torch.clamp(torch.log(mask), min=-1e10)  # Calculate the inf_mask where mask values are 0 (invalid) and will have -inf after log transformation
            logits_active_piece_id += inf_mask

        # Concatenate all logits into a single tensor as required by RLLIB
        all_logits = torch.cat([
            logits_active_piece_id,         # selected_piece_index
            logits_active_piece_side_index, # selected_side_index
        ], dim=1)

        return all_logits, state


    def value_function(self):
        value = self.value_head(self.shared_layers_output)
        return value.squeeze(-1)  # Remove any unnecessary dimensions to match [batch_size]


####################################################################################################################################################
# Policies for multi-agent Hierarchical-Autorregressive RL - High and Low Level Agents. The difference between the policies is mostly on the masking
####################################################################################################################################################

class HighARModel(TorchModelV2, nn.Module):
    """
    Combined model for autoregressive actions, integrating both context encoding and action logits computation.

    Args:
        obs_space (Space): Observation space.
        action_space (Space): Action space.
        num_outputs (int): Number of output features.
        model_config (dict): Model configuration.
        name (str): Model name.
    """

    def __init__(self, obs_space: Space, action_space:Space, num_outputs: int, model_config: Dict[str,Any], name: str)-> None:
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
            initializer=normc_init_torch(0.01),
        )

        self._context = None
        self.masks = None

    def forward(self, input_dict:Dict[str, torch.Tensor], state: List[torch.Tenso], seq_lens: torch.Tensor)-> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass to encode observations into context vector and prepare masks.
        Does not compute action logits.
        Args:
            input_dict (dict): Input dictionary containing observations and masks.
            state (Tensor): RNN state.
            seq_lens (Tensor): Sequence lengths for RNNs.

        Returns: Tuple[Tensor, Tensor]: Encoded context vector and state.
        """

        obs = input_dict["obs"]#["features"]  # Extract features from the observations
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

        return self._context, state

    def compute_action_logits(self, ctx_input:torch.Tensor, a1_input:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits for actions a1 and a2, applying masks if provided.

        Args:
            ctx_input (Tensor): Context vector from observations (computed by forward pass).
            a1_input (Tensor): Input from the first action.

        Returns: Tuple[Tensor, Tensor]: Logits for actions a1 and a2.
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

    def value_function(self)-> torch.Tensor:
        """
        Compute and return the value function from the context vector.

        Returns:
            Tensor: Value function output.
        """
        return torch.reshape(self.value_branch(self._context), [-1])


class LowARModel(TorchModelV2, nn.Module):
    """
    Combined model for autoregressive actions, integrating both context encoding and action logits computation.

    Args:
        obs_space (Space): Observation space.
        action_space (Space): Action space.
        num_outputs (int): Number of output features.
        model_config (dict): Model configuration.
        name (str): Model name.
    """

    def __init__(self, obs_space: Space, action_space:Space, num_outputs: int, model_config: Dict[str,Any], name: str)-> None:
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
            initializer=normc_init_torch(0.01),
        )

        self._context = None
        self.masks = None

    def forward(self, input_dict:Dict[str, torch.Tensor], state: List[torch.Tenso], seq_lens: torch.Tensor)-> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass to encode observations into context vector and prepare masks.
        Does not compute action logits.
        Args:
            input_dict (dict): Input dictionary containing observations and masks.
            state (Tensor): RNN state.
            seq_lens (Tensor): Sequence lengths for RNNs.

        Returns: Tuple[Tensor, Tensor]: Encoded context vector and state.
        """

        obs = input_dict["obs"]#["features"]  # Extract features from the observations
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

        return self._context, state

    def compute_action_logits(self, ctx_input:torch.Tensor, a1_input:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits for actions a1 and a2, applying masks if provided.

        Args:
            ctx_input (Tensor): Context vector from observations (computed by forward pass).
            a1_input (Tensor): Input from the first action.

        Returns: Tuple[Tensor, Tensor]: Logits for actions a1 and a2.
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

    def value_function(self)-> torch.Tensor:
        """
        Compute and return the value function from the context vector.

        Returns:
            Tensor: Value function output.
        """
        return torch.reshape(self.value_branch(self._context), [-1])




####################################################################################################################################################
# # Register the models with RLlib - this makes them globally accessible - used by D_ppo_config.py
####################################################################################################################################################

ModelCatalog.register_custom_model("masked_action_model", CustomMaskedModel) # For single agent
ModelCatalog.register_custom_model("custom_model_high", CustomModelHigh)     # For HRL
ModelCatalog.register_custom_model("custom_model_low", CustomModelLow)       # For HRL
