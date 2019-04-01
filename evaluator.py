import os
import glob
import time
from envs import get_render_func, get_vec_normalize
from tqdm import trange
import torch
from time import sleep

from envs import make_vec_envs


class Evaluator():
    def __init__(self, **args):

        torch.set_num_threads(1)

        self.load_dir = args['load_dir']
        self.det = args['deterministic_evaluation']
        self.algorithm = args['algorithm']

        self.env_name = args['env_name']

        self.grayscale = args['grayscale']
        self.skip_frame = args['skip_frame']
        self.num_frame_stack = args['num_frame_stack']

        self.scale = args['reward_scaling']

        self.seed = args['seed']

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

        self.env = make_vec_envs(self.env_name, self.seed + 1000, 1,
                                 None, None, 'cpu',
                                 False, self.grayscale, self.skip_frame, self.scale, num_frame_stack=self.num_frame_stack)
        # Get a render function
        self.render_func = get_render_func(self.env)

        # We need to use the same statistics for normalization as used in training
        self.actor_critic, self.ob_rms = \
            torch.load(os.path.join(self.load_dir,
                                    self.algorithm, self.env_name + ".pt"), map_location='cpu')
        self.actor_critic.to('cpu')
        self.vec_norm = get_vec_normalize(self.env)
        if self.vec_norm is not None:
            self.vec_norm.eval()
            self.vec_norm.ob_rms = self.ob_rms

    def evaluate(self):
        print("Evaluating\n------")
        rewards = []
        for i in trange(10):
            start = time.time()
            obs = self.env.reset()
            total_reward = 0
            recurrent_hidden_states = torch.zeros(
                1, self.actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            if self.render_func is not None and self.env_name.find('MarLo') == -1:
                self.render_func('human')
            while True:
                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = self.actor_critic.act(
                        obs, recurrent_hidden_states, masks, deterministic=self.det)

                # Obser reward and next obs
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                sleep(1.0 / 60)  # Max 60fps
                masks.fill_(0.0 if done else 1.0)
                if done:
                    rewards.append(total_reward[0][0])
                    break

                if self.render_func is not None and self.env_name.find('MarLo') == -1:
                    self.render_func('human')
        for i, r in enumerate(rewards):
            print("Episode {}: {}".format(i + 1, r))
        self.env.close()
        sleep(1)
        print("Total elapsed time: %.2f minutes" %
              ((time.time() - start) / 60.0))
