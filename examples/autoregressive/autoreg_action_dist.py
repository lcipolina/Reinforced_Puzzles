''' Test to see whether I can run the merged model AND apply masks'''


from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()



class TorchAutoregressiveCategoricalDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        """Sample actions deterministically."""
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        a1_logits, a2_logits = torch.split(logits, self.model.num_actions, dim=-1)
        a1_dist = TorchCategorical(a1_logits)
        a1 = a1_dist.deterministic_sample()
        a2_dist = TorchCategorical(a2_logits)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
        return torch.stack([a1, a2], dim=-1)

    def sample(self):
        """Sample actions stochastically."""
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        a1_logits, a2_logits = torch.split(logits, self.model.num_actions, dim=-1)
        a1_dist = TorchCategorical(a1_logits)
        a1 = a1_dist.sample()
        a2_dist = TorchCategorical(a2_logits)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
        return torch.stack([a1, a2], dim=-1)

    def logp(self, actions):
        """Calculate log probability of given actions."""
        a1, a2 = actions[:, 0], actions[:, 1]
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        a1_logits, a2_logits = torch.split(logits, self.model.num_actions, dim=-1)
        return TorchCategorical(a1_logits).logp(a1) + TorchCategorical(a2_logits).logp(a2)

    def sampled_action_logp(self):
        """Return log probability of the sampled action."""
        return self._action_logp

    def entropy(self):
        """Calculate entropy of the action distribution."""
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        a1_logits, a2_logits = torch.split(logits, self.model.num_actions, dim=-1)
        a1_dist = TorchCategorical(a1_logits)
        a2_dist = TorchCategorical(a2_logits)
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        """Calculate KL divergence with another distribution."""
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        a1_logits, a2_logits = torch.split(logits, self.model.num_actions, dim=-1)
        other_logits, _ = other.model(input_dict, None, None)
        other_a1_logits, other_a2_logits = torch.split(other_logits, self.model.num_actions, dim=-1)
        a1_dist = TorchCategorical(a1_logits)
        other_a1_dist = TorchCategorical(other_a1_logits)
        a1_terms = a1_dist.kl(other_a1_dist)
        a1 = a1_dist.sample()
        a2_dist = TorchCategorical(a2_logits)
        other_a2_dist = TorchCategorical(other_a2_logits)
        a2_terms = a2_dist.kl(other_a2_dist)
        return a1_terms + a2_terms

    def _a1_distribution(self):
        """Return the distribution for action a1."""
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        a1_logits, _ = torch.split(logits, self.model.num_actions, dim=-1)
        return TorchCategorical(a1_logits)

    def _a2_distribution(self, a1):
        """Return the distribution for action a2 conditioned on a1."""
        input_dict = {"obs": self.inputs}
        logits, _ = self.model(input_dict, None, None)
        _, a2_logits = torch.split(logits, self.model.num_actions, dim=-1)
        return TorchCategorical(a2_logits)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Compute the required model output shape dynamically."""
        num_actions = sum(space.n for space in action_space)
        return num_actions * 2
