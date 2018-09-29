import torch.nn as nn
import torch.optim as optim


class A2C():
    def __init__(self,
                 actor_critic,
                 value_loss_coef=None,
                 entropy_coef=None,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 ):

        # Actor critic is the policy network
        self.actor_critic = actor_critic
        # Loss parameters
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        # Remember, obs is a (T,N ,-1) matrix
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        # We compress all the tensors to (T * N, -1), we abstract over the parallel env
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            # We just send the first hidden state
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        # Standart A2C Loss
        # Put the tensor back into (T, N, -1) shape
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        # Clip the gradient
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
