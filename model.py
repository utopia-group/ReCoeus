import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


EPS = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self, num_features, num_actions):
        super().__init__()

        self.num_features = num_features
        self.num_actions = num_actions

        layer_sizes = [126, 64]
        dropout_probs = [0.5, 0.75]
        self.network = nn.Sequential(
            nn.Linear(num_features, layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_probs[0]),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_probs[1]),
            nn.Linear(layer_sizes[1], num_actions),
            nn.Softmax(dim=-1)
        )

    def _expand_mask(self, mask):
        expanded_mask = [0 for x in range(self.num_actions)]
        for i in mask:
            expanded_mask[i] = 1
        return expanded_mask

    def predict(self, state, mask):
        action_probs = self.network(torch.FloatTensor(state))
        mask = torch.FloatTensor(self._expand_mask(mask))
        masked_probs = action_probs * mask
        # Guard against all-zero probabilities
        guard_probs = torch.full((self.num_actions,), EPS) * mask
        return masked_probs + guard_probs

    def predict_masked_normalized(self, state, mask):
        action_probs = self.network(torch.FloatTensor(state))
        mask = torch.ByteTensor(self._expand_mask(mask))
        masked_probs = torch.masked_select(action_probs, mask)
        # Guard against all-zero probabilities
        masked_probs += torch.full((len(masked_probs),), EPS)
        normalized_probs = masked_probs / masked_probs.sum()
        return normalized_probs

    def sample_action(self, state, mask):
        probs = self.predict(state, mask)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.item()

    def sample_action_with_log_probability(self, state, mask):
        probs = self.predict(state, mask)
        distribution = Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    @staticmethod
    def save(model, path):
        model_descriptor = {
            'num_features': model.num_features,
            'num_actions': model.num_actions,
            'network': model.state_dict()
        }
        torch.save(model_descriptor, path)

    @staticmethod
    def load(path):
        model_descriptor = torch.load(path)
        num_features = model_descriptor['num_features']
        num_actions = model_descriptor['num_actions']
        model = Policy(num_features, num_actions)
        model.load_state_dict(model_descriptor['network'])
        return model

    @staticmethod
    def load_for_eval(path):
        model = Policy.load(path)
        model.eval()
        return model
