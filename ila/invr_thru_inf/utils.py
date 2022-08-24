# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    # NOTE: I guess I embed a tiny bug on data-collection code, and the stored observations
    # end up contain an extra dimension: (256, 4, 1, 1024) rather than (256, 4, 1024).
    # So I just force squeeze everything, which usually is not an issue here
    # (nothing has necessary extra dimension).
    return tuple(
        torch.as_tensor(
            x,
            device=device,
            dtype=(torch.float32 if x.dtype in [np.float64, torch.float64] else None),
        )
        for x in xs
    )


def repeat_batch(tensor, batch_size, device):
    """Converts a pixel obs (C,H,W) to a batch (B,C,H,W) of given size"""
    tensor = torch.as_tensor(tensor, device=device)
    if len(tensor.shape) == 3:
        return tensor.repeat(batch_size, 1, 1, 1)
    elif len(tensor.shape) <= 2:
        return tensor.repeat(batch_size, 1)
    else:
        raise ValueError(f'input shape needs to be 2 or 3 dimensional. Invalid input shape: {tensor.shape}')


# NOTE: used in this dir
def get_distr_string(env_name, distr_types, intensity=None, difficulty=None):
    print('distr_types', distr_types)
    if 'dmcgen-color-hard' in distr_types:
        return 'dmcgen-color-hard'

    assert "Reacher" not in env_name and 'reacher' not in env_name, 'reacher-easy will cause an issue in string matching in this function'

    if difficulty is None:
        if "-medium-" in env_name:
            distr_lvl = "-medium"
        elif "-hard-" in env_name:
            distr_lvl = "-hard"
        elif "-easy-" in env_name:
            distr_lvl = "-easy"
        else:
            distr_lvl = ""
    else:
        assert difficulty in ['medium', 'hard', 'easy']
        distr_lvl = "-" + difficulty

    if distr_types:
        distr_str = "-".join(distr[:3] for distr in sorted(distr_types))
    else:
        distr_str = "none"

    if intensity is not None:
        distr_intensity = f"-intsty{intensity:.4f}"
    else:
        distr_intensity = ""

    return f"{distr_str}{distr_lvl}{distr_intensity}"


def get_gpumemory_info():
    from ml_logger import logger
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    out = f'[MiB] total memory: {t/1024/1024}\treserved: {r/1024/1024}\tallocated: {a/1024/1024}\tfree: {f/1024/1024}'
    return out


def get_cuda_variables():
    """Only for debug purposes"""
    import os
    return {name: val for name, val in os.environ.items() if name.startswith('CUDA')}


def verify_args(current_args, loaded_args):
    for key, val in current_args.items():
        if key in loaded_args:
            if loaded_args[key] != val:
                raise ValueError(
                    f"loaded_args and current_args do not match.\nloaded_args: {loaded_args}\ncurrent_args: {current_args}"
                )


def update_args(original_args, new_args):
    update_dict = new_args if isinstance(new_args, dict) else vars(new_args)
    for key in vars(original_args):
        if key in update_dict:
            setattr(original_args, key, update_dict[key])


# NOTE: This is used in the directory
def get_buffer_prefix(train_env, action_repeat, seed,
                      snapshot_prefix=None, distraction_types=None, distraction_intensity=None,
                      augment=False, policy_on_clean_buffer='random', policy_on_distr_buffer='random',
                      eval_env=None, latent_buffer=False, verbose=False,
                      verbose_with_key=None):
    from pathlib import Path
    import hashlib
    distraction_types = [] if distraction_types is None else distraction_types

    path = (
        Path(f"separate_buffers/{policy_on_clean_buffer}-{policy_on_distr_buffer}")
        / f"{'aug' if augment else 'nonaug'}"
        / f"{train_env.split(':')[-1][:-3].lower()}"
    )

    from ml_logger import RUN
    if RUN.debug:
        path = path / 'debug'

    if latent_buffer:
        assert eval_env is None
        pretrained_agent_hash = hashlib.sha1(snapshot_prefix.encode('utf-8')).hexdigest()[:10]
        return (
            path
            / "original"
            / f"actrep{action_repeat}"
            / pretrained_agent_hash
            / f"{seed}"
        )
    elif verbose:
        assert eval_env
        pretrained_agent_hash = hashlib.sha1(snapshot_prefix.encode('utf-8')).hexdigest()[:10]
        return (
            path
            / f'actrep{action_repeat}'
            / f"{get_distr_string(eval_env, distraction_types, distraction_intensity)}"
            / pretrained_agent_hash
            / f"{seed}"
        )
    elif verbose_with_key:
        assert eval_env
        assert snapshot_prefix
        pretrained_agent_hash = hashlib.sha1(snapshot_prefix.encode('utf-8')).hexdigest()[:10]
        return (
            path
            / verbose_with_key
            / f'actrep{action_repeat}'
            / f"{get_distr_string(eval_env, distraction_types, distraction_intensity)}"
            / pretrained_agent_hash
            / f"{seed}"
        )

    assert eval_env
    return (
        path
        / "target"
        / f'actrep{action_repeat}'
        / f"{get_distr_string(eval_env, distraction_types, distraction_intensity)}"
        / f"{seed}"
    )


def get_bc_buffer_prefix(task, action_repeat, seed, distraction_types=None, snapshot_prefix=None,
                         distraction_intensity=None, augment=False, difficulty=None,
                         sample_action=False, random_action_prob=0.0):
    from pathlib import Path
    import hashlib
    distraction_types = [] if distraction_types is None else distraction_types

    path = (
        Path(f"bc_buffers/")
        / f"{'aug' if augment else 'nonaug'}"
        / task
    )

    if snapshot_prefix:
        pretrained_agent_hash = hashlib.sha1(snapshot_prefix.encode('utf-8')).hexdigest()[:10]
    else:
        pretrained_agent_hash = 'random'
    return (
        path
        / f'actrep{action_repeat}'
        / f"{get_distr_string(task, distraction_types, distraction_intensity, difficulty)}"
        / pretrained_agent_hash
        / ("sample_act" if sample_action else "determ_act")
        / f"randact_prob_{random_action_prob}"
        / f"{seed}"
    )

def record_video(Adapt, agent, env, progress, video_prefix='videos', size=64, cherry_pick=None):
    """
    clean_env, distr_env --> only for evalutation purposes!
    """
    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    from copy import deepcopy
    from ml_logger import logger

    prev_step, step, total_reward = 0, 0, 0
    past_return = 0
    rewards = []
    for episode in range(Adapt.num_eval_episodes):
        eval_agent = deepcopy(agent)  # make a new copy
        obs = env.reset()
        frames = []
        done = False
        ep_reward = 0
        while not done:
            # todo: use gym.env.render('rgb_array') instead
            frames.append(env.render(height=size, width=size))

            with torch.no_grad(), eval_mode(eval_agent):
                action = eval_agent.act(obs, progress.step, eval_mode=True)
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            ep_reward += reward
            step += 1

        # Append the last frame
        frames.append(env.render(height=size, width=size))

        if cherry_pick is None:
            video_file = f"{video_prefix}/{episode:03d}.mp4"
            logger.print(f"Saving video to {video_file}\tsteps {step}")
            logger.save_video(frames, video_file)
            for i, frm in enumerate(frames):
                logger.save_image(frm, f"{video_prefix}_frames/frame-{i:03d}.png")
        else:
            if episode == 0 or (cherry_pick == "max" and past_return < ep_reward) or (
                    cherry_pick == "min" and past_return > ep_reward
            ):
                saved_frames = deepcopy(frames)
                past_return = ep_reward

    if cherry_pick:
        logger.print(f"Saving video to {video_prefix}/{cherry_pick}.mp4\treturn: {past_return}")
        logger.save_video(saved_frames, f"{video_prefix}/{cherry_pick}.mp4")
        for i, frm in enumerate(saved_frames):
            logger.save_image(frm, f"{video_prefix}_frames/frame-{i:03d}.png")

    logger.print('total_reward:', total_reward / Adapt.num_eval_episodes)
