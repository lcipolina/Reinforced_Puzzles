import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.torch_utils import FLOAT_MIN


''' Compute raw logits for each part of the action space and apply masks to the logits.
    RLLIB will then sample actions from the action space based on the logits.
'''

class CustomMaskedModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        #____________________________________________________________________________________________________________________
        # Define the network architecture: i.e. the order of operations in the forward pass and their sizes
        # - Shared layers and separate output layers for each part of the action space
        #____________________________________________________________________________________________________________________

        self.action_space = action_space  # Save action space as an instance variable

        # Define the shared network with custom hidden layer sizes
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Separate output layers for each categorical dimension of the action space
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


        logits_target_piece = self.logits_target_piece_and_side(self.shared_layers_output)
        logit_target_side   = self.logits_target_piece_and_side(self.shared_layers_output)

        '''
        piece_id = [1,2,3,4]
        lado_piece_id = [5,6,7,8]
        target_piece_and_side = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        '''


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