"""
These functions are used to vectorize a bunch of environments to have multiple processes runnning in parallel
"""
import os

import gym
import numpy as np
from skimage import transform
import torch
from gym.spaces.box import Box
import warnings  # This ignore all the warning messages that are normally printed because of skiimage
from kits.minecraft.marlo_parallel import MarloEnvMaker

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

warnings.filterwarnings('ignore')


def make_env(env_id, seed, rank, log_dir, allow_early_resets, grayscale, skip_frame, scale, marlo_env_maker=None):
    def _thunk():
        if env_id.find('Vizdoom') > -1:
            import kits.doom  # noqa: F401

        if marlo_env_maker is not None:
            print("Creating a MarloEnv")
            env = marlo_env_maker.make_env(env_id)
            # Restrict the action list
            env = MarloWrapper(env)
        else:
            env = gym.make(env_id)
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        # We still want our different agents to have different seeds, otherwise they will experience the same things
        env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        if is_atari:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            if not is_atari:
                # Wrap deepmind already greyscale + resize + skip frame
                env = SkipWrapper(env, repeat_count=skip_frame)
                env = PreprocessImage(env, grayscale=grayscale)
                env = RewardScaler(env, scale=scale)
            env = TransposeImage(env)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir,
                  device, allow_early_resets, grayscale, skip_frame, scale, num_frame_stack=None):

    marlo_env_maker = None
    if env_name.find('MarLo') > -1:
        marlo_env_maker = MarloEnvMaker(num_processes)

    envs = [make_env(env_name, seed, i, log_dir, allow_early_resets, grayscale, skip_frame, scale, marlo_env_maker=marlo_env_maker)
            for i in range(num_processes)]

    print("{} process launched".format(len(envs)))
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
        # envs = FakeSubprocVecEnv(envs)
    else:
        #envs = FakeSubprocVecEnv(envs)
        envs = DummyVecEnv(envs)

    # Only use vec normalize for non image based env
    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecBezos(envs, device)

    if num_frame_stack is not None:
        envs = VecBezosFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        print("Auto Frame Stacking activated")
        envs = VecBezosFrameStack(envs, 4, device)
    print("Observation space: ", envs.observation_space.shape)
    print("Action space: ", envs.action_space)
    return envs


class PreprocessImage(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, grayscale=False,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        obs_shape = self.observation_space.shape
        n_colors = 1 if self.grayscale else obs_shape[2]
        self.observation_space = Box(
            0.0, 1.0, [height, width, n_colors], dtype=self.observation_space.dtype)

    def observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = transform.resize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = img.astype('float32') / 255.
        return img


class TransposeImage(gym.ObservationWrapper):
    """
    This wrapper puts the third dimension first (stacked greyscale frames or rgb)
    We do that because the pytorch conv expect a (N, C, -1) Shape where C is the number of channels (n of stacked frames or 3 for rgb)
    """

    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecBezos(VecEnvWrapper):
    """
    We extend the VecEnvWrapper of baseline to make it consistent with our implementation
    """

    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecBezos, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        # Remember that our actions are vectors, we transform them back into scalar (or from matrix to vector for continuous control)
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class VecNormalize(VecNormalize_):
    """
    You should not use VecNormalize on pixels because most of the cnn policies already normalize images (dividing by 255), unless you only want to normalize reward (in that case, you should pass ob=False)
    However, VecNormalize can be really useful for everything that is not pixels (e.g. joint space or [x,y,z] coordinates). Also, you have to save the running average if you want to play with a trained agent afterwards.

    Vecnormalize computes a running average to estimate mean and std for observations and rewards.
    Small modification of the baseline wrapper to prevent the running mean/std to being updated during eval
    """

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var +
                                                             self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecBezosFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class MarloWrapper(gym.Wrapper):
    def __init__(self, env, allowed_action_list=[3, 4, 7, 8]):
        super(MarloWrapper, self).__init__(env)
        self.allowed_action_list = allowed_action_list
        self.action_space = gym.spaces.Discrete(len(allowed_action_list))

    def step(self, action):
        real_action = self.allowed_action_list[action]
        return self.env.step(real_action)

    def reset(self):
        return self.env.reset()


class SkipWrapper(gym.Wrapper):
    """
        Generic common frame skipping wrapper
        Will perform action for `x` additional steps
    """

    def __init__(self, env, repeat_count=4):
        super(SkipWrapper, self).__init__(env)
        self.repeat_count = repeat_count
        self.stepcount = 0

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.repeat_count + 1) and not done:
            self.stepcount += 1
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        if 'skip.stepcount' in info:
            raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking '
                                  'the SkipWrapper wrappers.')
        info['skip.stepcount'] = self.stepcount
        return obs, total_reward, done, info

    def reset(self):
        self.stepcount = 0
        return self.env.reset()


def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None
