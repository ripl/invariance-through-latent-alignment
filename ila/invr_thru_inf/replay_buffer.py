# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return episode['reward'].shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, replay_dir, buffer_size):
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True, parents=True)
        self._current_episode = defaultdict(list)
        self._preload()
        self._buffer_size = buffer_size
        self._buffer_counter = 0
        self._filled = False

    def __len__(self):
        return self._num_transitions

    def is_filled(self):
        return self._filled

    def add(self, obs, reward, done, action, discount, state=None):
        # NOTE: reward = 0, action = [0, ..., 0], discount = 1.0 at the very first transition

        if self._buffer_counter < self._buffer_size:
            self._buffer_counter += 1
            self._filled = False
        else:
            self._filled = True

        if state is not None:
            extra_kwargs = {'state': state}  # num_action_repeat x state_dim
        else:
            extra_kwargs = {}
        name2val = dict(observation=obs, reward=reward, action=action, discount=discount, **extra_kwargs)
        for name, val in name2val.items():
            # NOTE:
            # NG: reward.shape == (256,)
            # OK: reward.shape == (256, 1)
            if np.isscalar(val):
                dtype = np.float32 if name in ['reward', 'discount', 'action'] else self._get_dtype(val)
                val = np.full((1,), val, dtype)
            # assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[name].append(val)

        if done:
            episode = dict()
            for name, val in self._current_episode.items():
                cur_episode = self._current_episode[name]
                dtype = np.float32 if name in ['reward', 'discount', 'action'] else self._get_dtype(val)
                episode[name] = np.array(cur_episode, dtype)
            self._current_episode = defaultdict(list)
            eps_fn = self._store_episode(episode)
            return eps_fn

    def add_extra(self, dictionary):
        for key, val in dictionary.items():
            self._current_episode['extra__' + key] = val

    def _get_dtype(self, val):
        if isinstance(val, np.ndarray):
            return val.dtype
        elif isinstance(val, float):
            return np.float32
        elif isinstance(val, int):
            return np.int32

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)
        return eps_fn


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, use_sample_aligned=True):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

        self._sample = self._sample_aligned if use_sample_aligned else self._sample_original

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        """Load from eps_fn and add its episodes to replay buffer in FIFO fashion.
        This is called by self._try_fetch().
        """
        try:
            episode = load_episode(eps_fn)
        except FileNotFoundError:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)  # Remove the oldest file
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len

            # NOTE: This is where it loads & stores episode locally
            # This method returns False only when loading file fails
            if not self._store_episode(eps_fn):
                break

    def _sample_aligned(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx - 1]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx - 1])
        discount = np.ones_like(episode['discount'][idx - 1])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx - 1 + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx - 1 + i] * self._discount

        return (obs, action, reward, discount, next_obs)

    def _sample_original(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, use_sample_aligned=True):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            use_sample_aligned=use_sample_aligned)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader, iterable


class Replay:
    def __init__(self, buffer_dir, buffer_size=256, batch_size=256, num_workers=1, nstep=1,
                 discount=0.99, use_sample_aligned=True):
        # NOTE: num_workers need to match during saving to and loading from the replay
        # otherwise some episodes will not be read.
        import os
        from pathlib import Path

        # todo: find ways to support remote/s3
        buffer_dir = Path(buffer_dir)
        self.storage = ReplayBufferStorage(buffer_dir, buffer_size)

        self.loader, self._replay_buffer = make_replay_loader(
            buffer_dir, buffer_size,
            batch_size, num_workers,
            save_snapshot=True, nstep=nstep, discount=discount,
            use_sample_aligned=use_sample_aligned)
        self._iter = None
        self.buffer_dir = buffer_dir

    @property
    def iterator(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        return self._iter
