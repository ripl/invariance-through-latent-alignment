# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from gym.core import Wrapper

import numpy as np
from gym import ObservationWrapper
from gym.wrappers import FrameStack


class VstackWrapper(ObservationWrapper):
    """A wrapper to stack observations. It works with LazyFrame"""
    def __init__(self, env):
        from gym import spaces
        super(VstackWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=(self.observation_space.shape[0] * self.observation_space.shape[1], *self.observation_space.shape[2:])
        )
        # self.observation_space = self.observation_space.shape[0] * self.observation_space.shape[1]

    def observation(self, lazy_frames):
        return np.vstack(lazy_frames)

class FrameStackWithState(FrameStack):
    """ Slightly customized version of FrameStack.
    This stacks info['sim_state'] as well
    """

    def __init__(self, env, num_stack, lz4_compress=False, state_key='sim_state'):
        super().__init__(env, num_stack, lz4_compress)
        self.state_frames = deque(maxlen=num_stack)
        self._state_key = state_key

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        self.state_frames.append(info[self._state_key])
        info.update({self._state_key: list(self.state_frames)})
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        dmc_env = self.env.unwrapped
        state = dmc_env.env.physics.get_state()
        [self.frames.append(observation) for _ in range(self.num_stack)]
        [self.state_frames.append(state) for _ in range(self.num_stack)]
        return self._get_observation()

class PixelObservation(Wrapper):
    def __init__(self, env, width=84, height=84, mode='rgb'):
        from gym import spaces
        super().__init__(env)
        self.mode = mode
        self.width = width
        self.height = height
        # pixels = self.env.render(mode=mode, width=width, height=height)
        pixels = self._render_and_transpose()
        self.observation_space = spaces.Box(
                shape=pixels.shape, low=0, high=255, dtype=pixels.dtype
            )

    def _render_and_transpose(self):
        obs = self.env.render(mode=self.mode, width=self.width, height=self.height)
        obs = obs.transpose(2, 0, 1)  # hwc --> chw
        return obs

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._render_and_transpose()
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        obs = self._render_and_transpose()
        return obs


def get_fetch_env(env_name, n_substeps, render_kwargs):
    def render_interface(env, **kwargs):
        return env.render(mode='rgb', **kwargs)

    import gym
    env = gym.make(f'fetch:{env_name}', n_substeps=n_substeps, reward_type='dense')
    env = PixelObservation(env, width=render_kwargs['width'], height=render_kwargs['height'])

    # Set up camera view point
    env.render(mode=None)  # Necessary to instantiate env.viewer first.
    env.viewer.cam.distance = 0.8  # Default: 1.0
    env.viewer.cam.azimuth = 180
    env.viewer.cam.elevation = -25
    env.viewer.cam.lookat[2] += 0.1

    return env


def get_env(name, frame_stack, action_repeat, seed,
            distraction_config=('background', 'camera', 'color'),
            rails_wrapper=False, save_info_wrapper=False, intensity=None, size=84):

    import functools
    # Common render settings
    module, env_name = name.split(':', 1)
    domain_name = env_name.split('-')[0].lower()
    camera_id = 2 if env_name.startswith('Quadruped') else 0  # zoom in camera for quadruped

    render_kwargs = dict(height=size, width=size, camera_id=camera_id)

    if module == "fetch":
        import gym
        # NOTE: default action_repeat (== n_substeps) in the package is 20.
        env = get_fetch_env(env_name, action_repeat, render_kwargs)
        env = FrameStack(env, frame_stack)
        env = VstackWrapper(env)
        env.render = functools.partial(env.render, mode='rgb')
        return env

    if module == "fake":
        import os
        from ila.invr_thru_inf.etc.fake_ur5_env import UR5FakeFetch
        from ila.invr_thru_inf.config import Adapt
        episodes_dir = Adapt.ur5_traj_dir
        env = UR5FakeFetch(episodes_dir, int_action=("grid" in env_name))
        env = PixelObservation(env, width=render_kwargs['width'], height=render_kwargs['height'])
        env = FrameStack(env, frame_stack)
        env = VstackWrapper(env)
        return env

    # import distracting_control  # TODO: Can we remove this??
    import gym
    # NOTE: distraction_config controls the all distractio settings
    # default setttings for distraction_control are: ('background', 'camera', 'color').
    # extra setting: 'video-background' and 'dmcgen-color-hard'

    # Replace video-background with background and set dynamic=True
    if 'video-background' in distraction_config:
        distraction_config = list(distraction_config)
        distraction_config.remove('video-background')
        distraction_config.append('background')
        dynamic = True
    else:
        dynamic = False

    if 'dmcgen-color-hard' in distraction_config:
        assert len(distraction_config) == 1, 'dmcgen-color-hard cannot be used with other distractions'
        dmcgen_color_hard = True
    else:
        dmcgen_color_hard = False

    if intensity:
        assert env_name.endswith('intensity-v1')

    # Default gdc background path: $HOME/datasets/DAVIS/JPEGImages/480p/
    print('making environment:', module, name, distraction_config)

    # NOTE: name format:
    #   f'distracting_control:{domain_name.capitalize()}-{task_name}-{difficulty}-v1'
    #   f'dmc:{domain_name.capitalize()}-{task_name}-v1'
    extra_kwargs = {}
    if module == 'distracting_control':
        import os
        extra_kwargs.update(dict(
            background_data_path=os.environ.get("DC_BG_PATH", None),
            distraction_seed=seed,
            distraction_types=distraction_config,
            dynamic=dynamic,
            fix_distraction=True,
            intensity=intensity,
            disable_zoom=True,
            sample_from_edge=bool(intensity)
        ))

        # Our project-specific distraction configuration
        if intensity:
            extra_kwargs.update({'sample_from_edge': True})

            if "background" in distraction_config:
                # Get max value of ground-plane-alpha
                if domain_name == 'reacher':
                    max_ground_plane_alpha = 0.0
                elif domain_name in ['walker', 'cheetah']:
                    max_ground_plane_alpha = 1.0
                else:
                    max_ground_plane_alpha = 0.3

                # intensity decides background alpha and ground-plane alpha.
                assert 0 <= intensity <= 1
                extra_kwargs.update({'background_kwargs': {
                    'video_alpha': intensity,
                    'ground_plane_alpha': 1.0 * (1.0 - intensity) + max_ground_plane_alpha * intensity
                }})

    env = gym.make(name, from_pixels=True, frame_skip=action_repeat, channels_first=True,
                    **render_kwargs, **extra_kwargs)
    env.seed(seed)

    # Inject RailsWrapper
    if save_info_wrapper:
        from .collect_offline_data import SaveInfoWrapper
        env = SaveInfoWrapper(env)

    # Wrappers
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    env = FrameStackWithState(env, frame_stack)
    env = VstackWrapper(env)

    if rails_wrapper:
        from .collect_offline_data import RailsWrapper
        env = RailsWrapper(env)

    # HACK: video_recorder requires access to env.render
    env.physics = env.unwrapped.env.physics

    import functools
    env.render = functools.partial(env.render, camera_id=0)

    return env


def parse_env_name(env_name):
    module, env = env_name.split(':', 1)
    task = env.lower()[:-3]  # remove "-v1"
    if module == 'distracting_control':
        difficulty = task.split('-')[-1]
        extra_rstring = '-' + difficulty
        task = task[:-len(extra_rstring)]
    else:
        difficulty = None
    return module, task, difficulty
