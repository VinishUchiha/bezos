import copy
import os
import glob
import time
from collections import deque

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm

from algorithms.policy_based.A2C import A2C
from algorithms.policy_based.PPO import PPO

from envs import make_vec_envs
from networks import ActorCriticNetwork
from storage import RolloutStorage
from envs import get_vec_normalize


class Runner():
    def __init__(self, **args):
        cuda = not args['no_cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        print("Model running on device: {}".format(self.device))
        torch.set_num_threads(1)

        self.env_name = args['env_name']
        self.epochs = args['epochs']
        self.num_processes = args['num_processes']
        self.num_steps = args['num_steps']
        self.num_test_episodes = args['num_test_episodes']
        self.test_every_n_epochs = args['test_every_n_epochs']
        self.use_deterministic_policy_while_testing = args['use_deterministic_policy_while_testing']

        self.grayscale = args['grayscale']
        self.skip_frame = args['skip_frame']
        self.num_frame_stack = args['num_frame_stack']

        self.num_updates_per_epoch = args['num_updates_per_epoch']
        self.num_steps = args['num_steps']

        self.use_gae = args['use_gae']
        self.gamma = args['gamma']
        self.tau = args['tau']

        self.reward_scaling = args['reward_scaling']

        self.seed = args['seed']
        self.log_dir = args['log_dir']
        self.save_dir = args['save_dir']

        try:
            os.makedirs(args['log_dir'])
            files = glob.glob(os.path.join(args['log_dir'], '*.manifest.json'))
            for f in files:
                os.remove(f)
        except OSError:
            files = glob.glob(os.path.join(args['log_dir'], '*.monitor.csv'))
            for f in files:
                os.remove(f)

        self.eval_log_dir = args['log_dir'] + "_eval"

        try:
            os.makedirs(self.eval_log_dir)
        except OSError:
            files = glob.glob(os.path.join(self.eval_log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

        self.envs = make_vec_envs(self.env_name, self.seed, self.num_processes,
                                  self.gamma, self.log_dir, self.device, False, self.grayscale, self.skip_frame, self.reward_scaling, num_frame_stack=self.num_frame_stack)

        self.algorithm = args['algorithm']
        # Decreasing LR scheduler
        self.scheduler = None

        if self.algorithm == 'A2C':
            actor_critic = ActorCriticNetwork(self.envs.observation_space.shape, self.envs.action_space,
                                              base_kwargs=args['policy_parameters'])
            actor_critic.to(self.device)
            self.policy = actor_critic
            self.agent = A2C(actor_critic, **args['algorithm_parameters'])

        elif self.algorithm == 'PPO':
            if(args['decreasing_lr']):
                def lambdalr(epoch): return ((float(self.epochs - epoch)) / float(self.epochs) * args['algorithm_parameters']['lr'])  # noqa: E704
                actor_critic = ActorCriticNetwork(self.envs.observation_space.shape, self.envs.action_space,
                                                  base_kwargs=args['policy_parameters'])
                actor_critic.to(self.device)
                self.policy = actor_critic
                self.agent = PPO(actor_critic, lambdalr, **
                                 args['algorithm_parameters'])
                self.scheduler = self.agent.scheduler
            else:
                actor_critic = ActorCriticNetwork(self.envs.observation_space.shape, self.envs.action_space,
                                                  base_kwargs=args['policy_parameters'])
                actor_critic.to(self.device)
                self.policy = actor_critic
                self.agent = PPO(actor_critic, None, **
                                 args['algorithm_parameters'])

        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                                       self.envs.observation_space.shape, self.envs.action_space,
                                       actor_critic.recurrent_hidden_state_size)
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        self.episode_rewards = deque(maxlen=50)
        self.writer = SummaryWriter(
            comment="{}-{}".format(self.env_name, self.algorithm))

    def run(self):
        start = time.time()
        for epoch in range(self.epochs):
            value_losses, action_losses, dist_entropies = [], [], []
            print("\nEpoch %d\n-------" % (epoch + 1))
            for j in trange(self.num_updates_per_epoch, leave=False):
                for step in range(self.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
                            self.rollouts.obs[step],
                            self.rollouts.recurrent_hidden_states[step],
                            self.rollouts.masks[step])

                    # Observe reward and next obs
                    obs, reward, done, infos = self.envs.step(action)
                    for info in infos:
                        if 'episode' in info.keys():
                            print("New episode")
                            self.episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                               for done_ in done])
                    self.rollouts.insert(obs, recurrent_hidden_states,
                                         action, action_log_prob, value, reward, masks)

                with torch.no_grad():
                    next_value = self.policy.get_value(self.rollouts.obs[-1],
                                                       self.rollouts.recurrent_hidden_states[-1],
                                                       self.rollouts.masks[-1]).detach()

                self.rollouts.compute_returns(
                    next_value, self.use_gae, self.gamma, self.tau)
                value_loss, action_loss, dist_entropy = self.agent.update(
                    self.rollouts)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)

                self.rollouts.after_update()

                total_num_steps = (epoch + 1) * (j + 1) * \
                    self.num_processes * self.num_steps

            end = time.time()
            print("Total timesteps: {}, FPS: {}".format(
                total_num_steps, int(total_num_steps / (end - start))))
            print("Statistic of the last %d episodes played" %
                  len(self.episode_rewards))
            if(len(self.episode_rewards) < 1):
                self.episode_rewards.append(0)
            episode_rewards_np = np.array(self.episode_rewards)
            value_losses = np.array(value_losses)
            action_losses = np.array(action_losses)
            dist_entropies = np.array(dist_entropies)
            print("Mean value loss: {}, Mean action loss: {}, Mean entropy: {}".format(
                value_losses.mean(), action_losses.mean(), dist_entropies.mean()))
            print(episode_rewards_np)
            print("Results: mean: {} +/- {}".format(np.mean(episode_rewards_np), np.std(episode_rewards_np)))
            print("Min: {}, Max: {}, Median: {}".format(np.min(episode_rewards_np), np.max(episode_rewards_np), np.median(episode_rewards_np)))

            self.writer.add_scalar(
                'value_loss/mean', value_losses.mean(), epoch)
            self.writer.add_scalar(
                'action_loss/mean', action_losses.mean(), epoch)
            self.writer.add_scalar(
                'dist_entropy/mean', dist_entropies.mean(), epoch)
            self.writer.add_scalar(
                'reward/mean', episode_rewards_np.mean(), epoch)
            self.writer.add_scalar(
                'reward/max', episode_rewards_np.max(), epoch)
            self.writer.add_scalar(
                'reward/min', episode_rewards_np.min(), epoch)

            if (epoch + 1) % self.test_every_n_epochs == 0:
                print("\nTesting...")
                bar = tqdm(total=self.num_test_episodes, leave=False)
                eval_envs = make_vec_envs(self.env_name, self.seed + self.num_processes,
                                          self.num_processes, self.gamma, self.eval_log_dir,
                                          self.device,
                                          True,
                                          self.grayscale, self.skip_frame, self.reward_scaling, num_frame_stack=self.num_frame_stack)
                vec_norm = get_vec_normalize(eval_envs)
                if vec_norm is not None:
                    vec_norm.eval()
                    vec_norm.ob_rms = get_vec_normalize(self.envs).ob_rm
                eval_episode_rewards = []
                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(self.num_processes,
                                                           self.policy.recurrent_hidden_state_size, device=self.device)
                eval_masks = torch.zeros(
                    self.num_processes, 1, device=self.device)

                while len(eval_episode_rewards) < self.num_test_episodes:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = self.policy.act(
                            obs, eval_recurrent_hidden_states, eval_masks, deterministic=self.use_deterministic_policy_while_testing)
                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)
                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                    for done_ in done])

                    for info in infos:
                        if 'episode' in info.keys():
                            bar.update(1)
                            eval_episode_rewards.append(
                                info['episode']['r'])
                eval_envs.close()
                bar.close()
                print(eval_episode_rewards)
                print(" Evaluation using {} episodes: mean reward {:.5f}, min/max {}/{}\n".
                      format(len(eval_episode_rewards),
                             np.mean(eval_episode_rewards), np.min(eval_episode_rewards), np.max(eval_episode_rewards)))

            print("Total elapsed time: %.2f minutes" %
                  ((time.time() - start) / 60.0))
            if self.scheduler is not None:
                print("Decreasing the learning rate...")
                self.scheduler.step()

            print("Saving the model...")
            save_path = os.path.join(self.save_dir, self.algorithm)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            save_model = self.policy
            if self.device == "cuda:0":
                save_model = copy.deepcopy(self.policy).cpu()
            save_model = [save_model,
                          getattr(get_vec_normalize(self.envs), 'ob_rms', None)]
            torch.save(save_model, os.path.join(
                save_path, self.env_name + ".pt"))
