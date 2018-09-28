import torch
import torch.nn as nn

from torch.distributions import Categorical
from initialisation import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NeuralCategorical(nn.Module):
    def __init__(self, num_features, num_outputs):
        # Fully connected layer mapping features to logits
        self.fc = nn.Linear(num_features, num_outputs)
        # Init the weights and biases
        init(self.fc, nn.init.orthogonal_,
             lambda b: nn.init.constant_(b, 0), gain=0.01)

    def forward(self, features):
        logits = self.fc(features)
        return Categorical(logits=logits)


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(ActorCriticNetwork, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.base = CNN(obs_shape[0], **base_kwargs)

        def init_(m):
            return init(m,
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(
            nn.Linear(base_kwargs.get(['hidden_size'], 512), 1))

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = NeuralCategorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(features)

        dist = self.dist(features)

        # We want action_log_probs and action to be a vector, not a scalar (which is the default output of torch.Categorical)
        if deterministic:
            action = dist.mode().unsqueeze(-1)
        else:
            action = dist.sample().unsqueeze(-1)

        action_log_probs = dist.log_probs(action.squeeze(-1)).unsqueeze(-1)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        features, _ = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.critic_linear(features)

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    """
    Base class of a (recurrent/non-recurrent) network which creates features based on the observation
    """

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            # We get a nice batch of xs and hidden states
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNN(NNBase):
    """
    Feature network using CNN
    """

    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNN, self).__init__(recurrent, hidden_size, hidden_size)

        def init_(m):
            return init(m,
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        features = self.main(inputs / 255.0)

        if self.is_recurrent:
            features, rnn_hxs = self._forward_gru(features, rnn_hxs, masks)

        return features, rnn_hxs
