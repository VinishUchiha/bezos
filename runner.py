import copy
import os
import glob
import time
from collections import deque

import numpy as np
import torch
from tqdm import trange, tqdm

from algorithms.policy_based.A2C import A2C
from envs import make_vec_envs
from networks import ActorCriticNetwork
from storage import RolloutStorage
from envs import get_vec_normalize


class Runner():
    def __init__(self, **args):
        cuda = not args['no_cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        torch.set_num_threads(1)

        self.env_name = args['env_name']
        self.epochs = args['epochs']
        self.num_processes = args['num_processes']
        self.num_steps = args['num_steps']
        self.num_test_episodes = args['num_test_episodes']
        self.test_every_n_epochs = args['test_every_n_epochs']

        self.grayscale = args['grayscale']
        self.skip_frame = args['skip_frame']
        self.num_frame_stack = args['num_frame_stack']

        self.num_updates_per_epoch = args['num_updates_per_epoch']
        self.num_steps = args['num_steps']

        self.use_gae = args['use_gae']
        self.gamma = args['gamma']
        self.tau = args['tau']

        self.seed = args['seed']
        self.log_dir = args['log_dir']
        self.save_interval = args['save_interval']
        self.save_dir = args['save_dir']

        try:
            os.makedirs(args['log_dir'])
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
                                  self.gamma, self.log_dir, self.device, False, self.grayscale, self.skip_frame, num_frame_stack=self.num_frame_stack)

        self.algorithm = args['algorithm']
        if self.algorithm == 'A2C':
            actor_critic = ActorCriticNetwork(self.envs.observation_space.shape, self.envs.action_space,
                                              base_kwargs=args['policy_parameters'])
            actor_critic.to(self.device)
            self.policy = actor_critic
            self.agent = A2C(actor_critic, **args['algorithm_parameters'])

        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                                       self.envs.observation_space.shape, self.envs.action_space,
                                       actor_critic.recurrent_hidden_state_size)
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        self.episode_rewards = deque(maxlen=40)

    def run(self):
        start = time.time()
        for epoch in range(self.epochs):
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

                self.rollouts.after_update()

                if j % self.save_interval == 0 and self.save_dir != "":
                    save_path = os.path.join(self.save_dir, self.algorithm)
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass

                    # A really ugly way to save a model to CPU
                    save_model = self.policy
                    if self.device == "cuda:0":
                        save_model = copy.deepcopy(self.policy).cpu()

                    save_model = [save_model,
                                  getattr(get_vec_normalize(self.envs), 'ob_rms', None)]

                    torch.save(save_model, os.path.join(
                        save_path, self.env_name + ".pt"))

                total_num_steps = (epoch + 1) * (j + 1) * \
                    self.num_processes * self.num_steps

            end = time.time()
            print("Total timesteps: {}, FPS: {}".format(
                total_num_steps, int(total_num_steps / (end - start))))
            print("Statistic of the last %d episodes played" %
                  len(self.episode_rewards))
            episode_rewards_np = np.array(self.episode_rewards)
            print("Results: mean: %.1f +/- %.1f," % (episode_rewards_np.mean(), episode_rewards_np.std()),
                  "min: %.1f," % episode_rewards_np.min(), "max: %.1f," % episode_rewards_np.max())
            if epoch % self.test_every_n_epochs == 0:
                print("\nTesting...")
                bar = tqdm(total=self.num_test_episodes)
                eval_envs = make_vec_envs(self.env_name, self.seed + self.num_processes,
                                          self.num_processes, self.gamma, self.eval_log_dir, self.device, True)
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
                            obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
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
                print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                      format(len(eval_episode_rewards),
                             np.mean(eval_episode_rewards)))

            print("Total elapsed time: %.2f minutes" %
                  ((time.time() - start) / 60.0))