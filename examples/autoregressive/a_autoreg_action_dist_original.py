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
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)  # prob of joint action LogP(a1, a2) = LogP(a1) + LogP(a2|a1)

        # Return the action tuple.
        return (a1, a2)

    def sample(self):
        """Sample actions stochastically."""
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def logp(self, actions):
        """Calculate log probability of given actions."""
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = torch.unsqueeze(a1.float(), 1)
        a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        return TorchCategorical(a1_logits).logp(a1) + TorchCategorical(a2_logits).logp(
            a2
        )

    def sampled_action_logp(self):
        """Return log probability of the sampled action."""
        return self._action_logp  # used for entropy, KL and gradient update

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

    # OBS: both _a1_distribution and _a2_distribution use self.model.action_module  - then they discard what they dont need (lazy programming)

    def _a1_distribution(self):
        """Return the distribution for action a1."""
        BATCH = self.inputs.shape[0]
        zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)       # Vector of zeros since a1 is not conditioned on anything
        # Uses the already initialized action_module
        a1_logits, _ = self.model.action_module(self.inputs, zeros)  # P(a1 | obs) , obs are the inputs: context vector and zero vector. Shorthand for: self.model.action_module.forward(self.inputs, zeros)
        a1_dist = TorchCategorical(a1_logits)                        # Returns a1 distribution
        return a1_dist

    def _a2_distribution(self, a1):
        """Return the distribution for action a2 conditioned on a1."""
        a1_vec = torch.unsqueeze(a1.float(), 1)                       # Input for the distrib of a2 is a1 plus the context vector
        # Uses the already initialized action_module
        _, a2_logits = self.model.action_module(self.inputs, a1_vec)  # Calls the action_module to get logits for a2. Shorthand for: self.model.action_module.forward(self.inputs, zeros)
        a2_dist = TorchCategorical(a2_logits)                         # Create a categorical distribution for a2
        return a2_dist

    @staticmethod
    def required_model_output_shape_old(action_space, model_config):
        '''This one came with the exaplme from RLLIB'''
        return 16  # controls model output feature vector size

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Compute the required model output shape dynamically."""
        # Compute the required model output shape dynamically
        # Size of the hidden layers
        num_actions = sum(space.n for space in action_space)
        return num_actions*2  # Return the total number of actions