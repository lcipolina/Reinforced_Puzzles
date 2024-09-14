'''THIS WORKS - splits in 2 the logit calculation"
'''
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
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
        return (a1, a2)

    def sample(self):
        """Sample actions stochastically."""
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
        return (a1, a2)

    def logp(self, actions):
        """Calculate log probability of given actions."""
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = torch.unsqueeze(a1.float(), 1)
        a1_logits, a2_logits = self.model.compute_action_logits(self.inputs, a1_vec)
        return TorchCategorical(a1_logits).logp(a1) + TorchCategorical(a2_logits).logp(a2)

    def sampled_action_logp(self):
        """Return log probability of the sampled action."""
        return self._action_logp

    def entropy(self):
        """Calculate entropy of the action distribution."""
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        """Calculate KL divergence with another distribution."""
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())
        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a1_distribution(self):
        """Return the distribution for action a1."""
        BATCH = self.inputs.shape[0]
        zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
        a1_logits, _ = self.model.compute_action_logits(self.inputs, zeros)
        a1_dist = TorchCategorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        """Return the distribution for action a2 conditioned on a1."""
        a1_vec = torch.unsqueeze(a1.float(), 1)
        _, a2_logits = self.model.compute_action_logits(self.inputs, a1_vec)
        a2_dist = TorchCategorical(a2_logits)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Compute the required model output shape dynamically."""
        num_actions = sum(space.n for space in action_space)
        return num_actions * 2
