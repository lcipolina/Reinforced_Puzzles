from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedActionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MaskedActionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Access sub-spaces within the observation space and exclude the action mask
        original_spaces = obs_space.original_space.spaces
        flattened_size = sum(
            torch.prod(torch.tensor(space.shape)).item()  # Multiply dimensions of each feature to get total size
            for key, space in original_spaces.items()
            if key != "action_mask"  # Skip the action mask
        )

        # Define layers based on the calculated flattened size
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.value_branch = nn.Linear(128, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Extract the action mask, which has the batch size as the first dimension
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.float32)

        # Flatten all non-mask features for each observation and concatenate across features
        # This will be the input to the first layer
        # For example, if each feature set is (32, 16) and there are two such feature sets, the final concatenated shape will be (32, 32).
        non_mask_features = torch.cat([
            torch.tensor(value, dtype=torch.float32).view(action_mask.size(0), -1)  #convert to tensor and reshape to (batch_size, -1) for concatenation.
            for key, value in obs.items()
            if key != "action_mask" # Skip the action mask
        ], dim=1)

        # Debugging: Verify the shape of non_mask_features
        print(f"Shape of non_mask_features: {non_mask_features.shape}")

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
        return self._value_out.squeeze(1)
