from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedActionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MaskedActionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Flatten the observation space to pass through a fully connected layer
        # Calculate the correct input sizes for the fully connected layers

        attribute_name = "original_space"                         # (same as obs_space.original_space but with dynamic access)
        obs_space = getattr(obs_space, attribute_name, obs_space) # dict
        current_puzzle_size = obs_space["current_puzzle"].shape[0] * obs_space["current_puzzle"].shape[1]
        available_pieces_size = obs_space["available_pieces"].shape[0] * obs_space["available_pieces"].shape[1]
        available_connections_size = obs_space["available_connections"].shape[0]

        # Define the neural network layers
        self.fc_current_puzzle = nn.Linear(current_puzzle_size, 128)
        self.fc_available_pieces = nn.Linear(available_pieces_size, 128)
        self.fc_available_connections = nn.Linear(available_connections_size, 128)

        # Define separate output layers for each component of the action space
        self.fc_piece_id = nn.Linear(128 * 3, 4)
        self.fc_target_id = nn.Linear(128 * 3, 4)
        self.fc_side_index = nn.Linear(128 * 3, 4)
        self.fc_target_side_index = nn.Linear(128 * 3, 4)

        '''
        # Access sub-spaces within the observation space and exclude the action mask
        #TODO: use this to create the layers automatically - this gives an 'orderedDict' which is no
        original_spaces = obs_space.original_space.spaces  #orderedDict
        flattened_size = sum(
            torch.prod(torch.tensor(space.shape)).item()  # Multiply dimensions of each feature to get total size
            for key, space in original_spaces.items()
            if key != "action_mask"  # Skip the action mask
        )

        # Define layers based on the calculated flattened size
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.value_branch = nn.Linear(128, 1)
        '''
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Flatten the components of the observation dictionary
        current_puzzle_flat = torch.flatten(obs["current_puzzle"], start_dim=1)
        available_pieces_flat = torch.flatten(obs["available_pieces"], start_dim=1)
        available_connections_flat = obs["available_connections"].float()

        # Pass through the respective layers
        current_puzzle_out = torch.relu(self.fc_current_puzzle(current_puzzle_flat))
        available_pieces_out = torch.relu(self.fc_available_pieces(available_pieces_flat))
        available_connections_out = torch.relu(self.fc_available_connections(available_connections_flat))

        # Concatenate the outputs
        combined = torch.cat([current_puzzle_out, available_pieces_out, available_connections_out], dim=1)

        # Compute logits for each component
        logits_piece_id = self.fc_piece_id(combined)
        logits_target_id = self.fc_target_id(combined)
        logits_side_index = self.fc_side_index(combined)
        logits_target_side_index = self.fc_target_side_index(combined)

        # Reshape action mask to match logits shape for each component
        # Reshape action mask to match logits shape for each component
        action_mask = obs["action_mask"].float().reshape(-1, 256)
        mask_piece_id = action_mask[:, :4]
        mask_target_id = action_mask[:, 4:8]
        mask_side_index = action_mask[:, 8:12]
        mask_target_side_index = action_mask[:, 12:16]

        # Apply the action mask by setting invalid action logits to a very negative value
        value = -1e6  # TODO: remove this line - this is to test the model without the mask

        masked_logits_piece_id = logits_piece_id + (mask_piece_id * value)
        masked_logits_target_id = logits_target_id + (mask_target_id * value)
        masked_logits_side_index = logits_side_index + (mask_side_index * value)
        masked_logits_target_side_index = logits_target_side_index + (mask_target_side_index * value)

        return torch.cat([
            masked_logits_piece_id,
            masked_logits_target_id,
            masked_logits_side_index,
            masked_logits_target_side_index
        ], dim=1), state


    def forward_old(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Extract the action mask, which has the batch size as the first dimension
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.float32)

        # Flatten all non-mask features for each observation and concatenate across features
        # This will be the input to the first layer
        # For example, if each feature set is (batch size, num dims) (32, 16) and there are two such feature sets, the final concatenated shape will be (32, 32).
        non_mask_features = torch.cat([
            torch.tensor(value, dtype=torch.float32).view(action_mask.size(0), -1)  #convert to tensor and reshape to (batch_size, -1) for concatenation.
            for key, value in obs.items()
            if key != "action_mask" # Skip the action mask
        ], dim=1)

        # Pass the batch of non-mask features through the first layer
        x = F.relu(self.fc1(non_mask_features))
        logits = self.fc2(x)
        self._value_out = self.value_branch(x)

        # Apply the action mask to the logits
        inf_mask = torch.clamp(torch.log(action_mask + 1e-6), min=-1e10)

        if inf_mask.shape == logits.shape:
            logits += inf_mask

        return logits, state

    def value_function(self):
        # Return the predicted value function from the value branch
        return torch.zeros(1)  # Dummy value for the sake of completeness
