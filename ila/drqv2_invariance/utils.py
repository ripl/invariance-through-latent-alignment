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
    return tuple(
        torch.as_tensor(
            x,
            device=device,
            dtype=(torch.float32 if x.dtype in [np.float64, torch.float64] else None),
        )
        for x in xs
    )


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def get_distr_string(env_name, distr_types, intensity=None):
    print('distr_types', distr_types)
    if distr_types is None:
        return "none"

    if 'dmcgen-color-hard' in distr_types:
        return 'dmcgen-color-hard'

    if "-medium-" in env_name:
        distr_lvl = "-med"
    elif "-hard-" in env_name:
        distr_lvl = "-hard"
    else:
        distr_lvl = ""
    distr_str = "-".join(distr[:3] for distr in sorted(distr_types))

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
    out = f'total memory: {t/1000}\treserved: {r/1000}\tallocated: {a/1000}\tfree: {f/1000}'
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


def visualize_buffer_episodes(replay, key='video/buffer-viz.mp4'):

    def get_img(obs):
        """Convert stacked observation to image.

        This expects (channel, width, height) format
        """
        assert obs.shape[0] % 3 == 0
        imgs = [obs[i*3:(i+1)*3, :, :] for i in range(int(obs.shape[0] / 3))]
        # imgs = np.concatenate(imgs, axis=2)  # Stack images w.r.t. width
        imgs = torch.cat(imgs, dim=2)  # Stack images w.r.t. width
        # return imgs.transpose(1, 2, 0)
        return imgs.permute(1, 2, 0).cpu().numpy()

    from ml_logger import logger
    from .config import Args
    batch = next(replay.iterator)
    batch_obs = to_torch(batch, Args.device)[0]  # (256, 9, 84, 84)
    logger.print('vis batch_obs shape', batch_obs.shape)
    logger.print('vis batch_obs dtype', batch_obs.dtype)
    frames = [get_img(obs) for obs in batch_obs]
    logger.save_video(frames, key)


def set_egl_id():
    # NOTE: CUDA_VISIBLE_DEVICES is set a little late by the node. Thus, putting
    # export EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES does NOT work. Thus I do it here manually.
    from ml_logger import logger
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES', False) and 'EGL_DEVICE_ID' not in os.environ:
        logger.print('pre EGL_DEVICE_ID', os.environ.get('EGL_DEVICE_ID', 'variable not found'))
        cvd = os.environ['CUDA_VISIBLE_DEVICES']
        if ',' in cvd:
            cvd = cvd.split(',')[0]
        os.environ['EGL_DEVICE_ID'] = cvd
        logger.print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'variable not found'))
        logger.print('EGL_DEVICE_ID', os.environ.get('EGL_DEVICE_ID', 'variable not found'))


def update_args(original_args, new_args):
    update_dict = new_args if isinstance(new_args, dict) else vars(new_args)
    for key in vars(original_args):
        if key in update_dict:
            setattr(original_args, key, update_dict[key])
