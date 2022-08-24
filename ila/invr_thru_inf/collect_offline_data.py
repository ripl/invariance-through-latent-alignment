#!/usr/bin/env python3

from pathlib import Path
import torch
from gym import Wrapper
from . import utils
from ila.helpers import logger, mllogger
import numpy as np
import math

import wandb


def upload_and_cleanup(buffer_dir, target_path, archive):
    import shutil
    # Upload target-obs-buffer
    mllogger.start('uploading_tar')
    mllogger.upload_dir(buffer_dir, target_path, archive='tar')
    elapsed = mllogger.since('uploading_tar')
    logger.print(f'Upload took {elapsed} seconds')

    # local cleanup
    if buffer_dir.is_dir():  # TEMP
        logger.print('Removing the local buffer and archive...')
        shutil.rmtree(buffer_dir)


class SaveInfoWrapper(Wrapper):
    """Nothing but save info on clean environment"""
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state = self.env.unwrapped.env.physics.get_state()

        # That this assertion holds actually means this wrapper is not necessary
        # But this is clearer since how / when the state is saved is explicit.
        assert (info['sim_state'] == state).all(), f"{info['sim_state']}\nvs\n{state}"

        # Overwrite sim_state
        info = {'sim_state': state}
        return obs, reward, done, info


class RailsWrapper(Wrapper):
    """This wrapper is used to replay (reproduce) an episode.

    This is injected to wrap DMEnv or DistractingEnv
    and completly fakes the output of step() and reset() so that the trajectory
    replays exactly what is registered in self._episode.
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._episode = None
        self._counter = 0
        self._state_sequence = []

    def _get_obs_pixels(self):
        from distracting_control.gym_env import DistractingEnv
        from gym_dmc.dmc_env import DMCEnv

        env = self.env.unwrapped
        if isinstance(env, DMCEnv):
            obs = env._get_obs_pixels()
        elif isinstance(env, DistractingEnv):
            pixel_wrapper = env.env
            # obs = dmenv._task.get_observation(dmenv._physics)  # This is for non-pixel
            obs = pixel_wrapper.physics.render(**pixel_wrapper._render_kwargs)
            # pixels = dmenv._env.render(**self._render_kwargs)
            if self.env.channels_first:
                obs = obs.transpose([2, 0, 1])
        else:
            raise ValueError('The env under RailsWrapper is invalid:', self.env.unwrapped)
        return obs

    def reset(self):
        assert self._action_sequence is not None
        assert self._reward_sequence is not None

        assert self._episode, 'self._episode is empty. self.set_episode(episode) must be called before reset.'
        self._counter = 0
        self._rew_error = []
        self._state_error = []

        # self.env.set_state(self._initial_state)
        suite_env = self.env.unwrapped.env

        # NOTE: This is not ideal (action: [0] hasn't been executed)
        # Simulate the initial obs by stepping with [0] action!
        self.env.reset()
        suite_env.physics.reset()  # Critical for reproduction. (https://github.com/deepmind/dm_control/issues/64)
        assert self._state_sequence[0].dtype == np.float64
        suite_env.physics.set_state(self._state_sequence[0])  # NOTE: this is in replacement of env.set_state!!!
        obs, *_ = self.env.step([0])  # Mimic the initial observation

        # Reset simulation state again
        suite_env.physics.reset()
        suite_env.physics.set_state(self._state_sequence[0])
        return obs

    def step(self, action):
        """Step following the rail.

        action is ignored.
        """
        assert self._action_sequence is not None
        assert self._reward_sequence is not None
        assert action is None, 'Set action=None as RailsWrapper ignores it.'

        act = self._action_sequence[self._counter]
        assert act.dtype == np.float32
        next_obs, rew, _, _ = self.env.step(act)

        # For debug:
        # import math
        # self._rew_error.append(math.sqrt((rew - self._reward_sequence[self._counter]) ** 2))
        # if self._counter < len(self._state_sequence) - 1:
        #     error = np.sqrt((self.env.unwrapped.env.physics.get_state() - self._state_sequence[self._counter + 1]) ** 2).mean()
        #     self._state_error.append(error)
        #     logger.print(f'state error: {error}')
        # print('rew_error', self._rew_error[-1])
        # print('state_error', self._state_error[-1])

        self._counter += 1

        done = False
        if self._counter == len(self._state_sequence):  # -1 because we don't use the last element in self._state_sequence
            done = True
            self._episode = None
            self._action_sequence = None

            # For debug:
            # wandb.log({'reward_error': wandb.Histogram(self._rew_error),
            #            'state_error': wandb.Histogram(self._state_error)})

        # Overwrite info since self.env.step does not work properly.
        info = {'sim_state': self.env.unwrapped.env.physics.get_state()}
        return next_obs, rew, done, info

    def set_episode(self, episode, target=None):
        """Set the episode which RailsWrapper uses to replay.

        episode['state'][n] contains stacked states that correspond to frame-stacked observations.
        Always use the last element of the stacked state. That corresponds to the latest observation.
        """
        self._episode = episode

        # NOTE: during training, just after reset, all rows of stacked frames are identical.
        # episode['state'] contains sequence of stacked states only after the reset.
        # episode['state'][0] should look like [init_state, init_state, second_state]
        # episode['state'][1] would be         [init_state, second_state, third_state]
        # self._initial_state = episode['state'][0][0]
        # self._state_sequence = [self._initial_state] + [stacked_states[-1] for stacked_states in episode['state']]  # You don't use the last state.
        self._state_sequence = episode['state']
        self._action_sequence = episode['action']
        self._reward_sequence = episode['reward']
        assert len(self._state_sequence) == len(self._action_sequence) == len(self._reward_sequence)

        dm_env = self.env.unwrapped.env

        # TODO: This does not work!!!
        # it does not reflect the values set.
        # HOWEVER, as long as we align random seed on two environments, goal positions are sampled in a same way.
        if 'Reacher' in self.env.unwrapped.spec.id:
            dm_env.physics.named.model.geom_size['target', 0] = episode['extra__goal_size']
            dm_env.physics.named.model.geom_size['target', 'x'] = episode['extra__goal_x']
            dm_env.physics.named.model.geom_size['target', 'y'] = episode['extra__goal_y']


def collect_trajectories(agent, env, buffer_dir, num_steps, pretrain_last_step, replay,
                         random_policy=False,
                         encode_obs=False, augment=None, store_states=False,
                         action_repeat=2,
                         device='cuda', video_key='noname',
                         time_limit=None,
                         reset_physics=True):
    """Run agent in env, and collect trajectory.
    Save each transition as (latent, action, reward, next_latent) tuple,
    reusing the original ReplayBuffer.

    src_state_replay: specify a replay buffer whose episode contains "state" information.
    When this is set, this function assumes env is wrapped with RailsWrapper, and calls env.set_episode(episode)
    before env.reset()
    """
    from pathlib import Path

    logger.print('(beginning) num_steps', num_steps)
    logger.print('(beginning) action_repeat', action_repeat)

    buffer_dir = Path(buffer_dir)

    def manipulate_obs(observation):
        # Observation manipulation
        if augment:
            num_augments = 4
            # obs: (9, 84, 84)
            # batch_obs: (4, 9, 84, 84)
            batch_obs = utils.repeat_batch(observation, batch_size=num_augments, device=device)
            observation = augment(batch_obs.float())

        if encode_obs:
            observation = agent.encode(observation, to_numpy=True).squeeze()

        return observation

    def calc_remaining_steps(num_steps):
        if buffer_dir.is_dir():
            import math
            from .adapt import num_transitions_per_file
            num_files = len(list(buffer_dir.glob('*.npz')))
            logger.print(f'{num_files} files are found in {buffer_dir}')
            last_steps = (num_files * num_transitions_per_file) / action_repeat
            logger.print('num_steps', num_steps)
            logger.print('last_steps', last_steps)
            logger.print('action_rep', action_repeat)
            return num_steps - last_steps
        return num_steps

    remaining_steps = calc_remaining_steps(num_steps)
    assert remaining_steps >= 0, f'remaining_steps is negative!!: {remaining_steps}: {buffer_dir}'
    fill_buffer = remaining_steps > 0

    # Policy will not run if latent_buffer directory exists.
    if fill_buffer:
        mllogger.start('fill_buffer')
        logger.print(f"Needs to fill buffer: {buffer_dir}\nenvironment: {env}")
        logger.print(f'{remaining_steps} steps to go: {buffer_dir}')

        step = 0
        video_ep = 0
        num_video_episodes = 5
        while step < remaining_steps:

            obs = env.reset()
            if reset_physics:
                init_state = env.unwrapped.env.physics.get_state()

                # Critical for reproduction. (https://github.com/deepmind/dm_control/issues/64)
                env.unwrapped.env.physics.reset()  # Needs to reset AFTER env reset!!

                # Restore initial state
                env.unwrapped.env.physics.set_state(init_state)

            # Check timelimit!!
            if time_limit and mllogger.since('run') > time_limit:
                logger.print(f'local time_limit: {time_limit} (sec) has reached!')
                # Cleanup
                num_files = len(list(buffer_dir.iterdir()))
                logger.print(f'{num_files} files have been generated in this run!')
                raise TimeoutError

            frames = []
            done = False

            # NOTE: Needs to store the goal info since that's not in 'state'
            # if 'Reacher' in env.unwrapped.spec.id and store_states:
            #     print('saving extra info!!')
            #     goal_info = {
            #         'goal_size': env.physics.named.model.geom_size['target', 0],
            #         'goal_x': env.physics.named.model.geom_pos['target', 'x'],
            #         'goal_y': env.physics.named.model.geom_pos['target', 'y']
            #     }
            #     print('goal_info', goal_info)
            #     replay.storage.add_extra(goal_info)

            while not done:
                if video_ep < num_video_episodes:
                    frames.append(
                        env.render(height=256, width=256)
                    )

                if agent is None or random_policy:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad(), utils.eval_mode(agent):
                        action = agent.act(obs)

                action = action.astype(np.float32)  # Explicitly cast to float32 before env.step
                extra = {'state': env.unwrapped.env.physics.get_state()} if store_states else {}
                next_obs, reward, done, _ = env.step(action)

                # extra = {'state': info['sim_state']} if store_states else {}
                transition = dict(
                    obs=manipulate_obs(obs),
                    reward=reward,
                    done=done,
                    discount=1.0,
                    action=action,
                    **extra
                )

                eps_fn = replay.storage.add(**transition)
                obs = next_obs
                step += 1

            if frames:
                frames.append(env.render(height=256, width=256))
                frames = np.asarray([np.asarray(f) for f in frames], dtype=np.uint8)
                frames = np.asarray([np.asarray(f).transpose(2, 0, 1) for f in frames])
                wandb.log({video_key: wandb.Video(frames, fps=20, format='mp4')})
                video_ep += 1

        logger.print('num_steps', num_steps)
        assert (
            calc_remaining_steps(num_steps) == 0
        ), f"The number of files does not match!!: {buffer_dir}\ndetected remaining step: {calc_remaining_steps(num_steps)}"

    else:
        logger.print(f"Buffer is filled: {buffer_dir}")

        from .adapt import verify_local_buffer
        logger.print('num_steps', num_steps)
        if not verify_local_buffer(buffer_dir, num_steps * action_repeat):
            num_files = len(list(buffer_dir.iterdir()))
            raise RuntimeError(
                f'Donwloaded buffer contains {num_files} files!! Aborting.'
            )

    return replay


def collect_trajectories_on_rails(agent, env, buffer_dir, num_steps, replay, src_state_replay,
                                  encode_obs=False, augment=None, action_repeat=2,
                                  device='cuda', video_key='noname',
                                  time_limit=None):
    from pathlib import Path
    buffer_dir = Path(buffer_dir)

    def manipulate_obs(observation):
        # Observation manipulation
        if augment:
            num_augments = 4
            # obs: (9, 84, 84)
            # batch_obs: (4, 9, 84, 84)
            batch_obs = utils.repeat_batch(observation, batch_size=num_augments, device=device)
            observation = augment(batch_obs.float())

        if encode_obs:
            observation = agent.encode(observation, to_numpy=True).squeeze()

        return observation

    def calc_remaining_steps(num_steps):
        if buffer_dir.is_dir():
            import math
            from .adapt import num_transitions_per_file
            num_files = len(list(buffer_dir.glob('*.npz')))
            logger.print(f'{num_files} files are found in {buffer_dir}')
            last_steps = (num_files * num_transitions_per_file) / action_repeat
            logger.print('num_steps', num_steps)
            logger.print('last_steps', last_steps)
            logger.print('action_rep', action_repeat)
            return num_steps - last_steps
        return num_steps

    remaining_steps = calc_remaining_steps(num_steps)
    assert remaining_steps >= 0, f'remaining_steps is negative!!: {remaining_steps}: {buffer_dir}'
    fill_buffer = remaining_steps > 0

    # Policy will not run if latent_buffer directory exists.
    if fill_buffer:
        mllogger.start('fill_buffer')
        logger.print(f"Needs to fill buffer: {buffer_dir}\nenvironment: {env}")
        logger.print(f'{remaining_steps} steps to go: {buffer_dir}')

        step = 0
        video_ep = 0
        num_video_episodes = 5

        # Get the reference to rails_env
        rails_env = env
        while not isinstance(rails_env, RailsWrapper):
            rails_env = rails_env.env

        while step < remaining_steps:
            if not src_state_replay._replay_buffer._episodes:
                # fetch episodes internally
                import time
                started = time.time()
                src_state_replay._replay_buffer._try_fetch()
                elapsed = time.time() - started
                logger.info(f'buffer._try_fetch() took {elapsed}')

            # TODO: Use all episodes one by one rather than random sample
            episode = src_state_replay._replay_buffer._sample_episode()
            # Env is RailsEnv that replays an episode from the buffer.
            # Sample an episode from src_state_buffer_dir, and env.set_episode(episode) before env.reset()
            env.set_episode(episode)

            obs = env.reset()
            # NOTE: Intentionally skip storin ghe first transition
            # since we don't have info['sim_state']
            # but this is handled. (Look at RailsWrapper._initial_state)

            # Check timelimit!!
            if time_limit and mllogger.since('run') > time_limit:
                logger.print(f'local time_limit: {time_limit} (sec) has reached!')
                # Cleanup
                num_files = len(list(buffer_dir.iterdir()))
                logger.print(f'{num_files} files have been generated in this run!')
                raise TimeoutError

            frames = []
            act_diffs = []
            done = False

            while not done:

                # Get original observation and calculate corresponding action
                orig_obs = episode['observation'][rails_env._counter]
                orig_act = episode['action'][rails_env._counter]
                with torch.no_grad(), utils.eval_mode(agent):
                    expert_action = agent.act(orig_obs)
                act_diff = np.linalg.norm(orig_act - expert_action)
                act_diffs.append(act_diff)

                # Action does not matter as it is rails env.
                next_obs, reward, done, _ = env.step(None)

                transition = dict(
                    obs=manipulate_obs(obs),
                    reward=reward,
                    done=done,
                    discount=1.0,
                    action=expert_action  # NOTE: store expert action rather than actual action taken!!
                )

                # Save video of obs and orig_obs
                if video_ep < num_video_episodes:
                    from ila.helpers.media import tile_stacked_images
                    img_obs = tile_stacked_images(obs)
                    img_orig_obs = tile_stacked_images(orig_obs)
                    stacked = np.concatenate([img_obs, img_orig_obs], axis=0)
                    frames.append(stacked)

                eps_fn = replay.storage.add(**transition)
                obs = next_obs
                step += 1

            # Visualize histogram on action difference
            wandb.Histogram(act_diffs)

            # Save video if frames is non-empty
            if frames:
                # frames = np.asarray([np.asarray(f) for f in frames], dtype=np.uint8)
                # logger.print('frame[0].shape', frames[0].shape)
                # mllogger.save_video(frames, key=f"videos/{video_key}/{eps_fn[:-4]}.mp4")
                frames = np.asarray([np.asarray(f).transpose(2, 0, 1) for f in frames])
                wandb.log({video_key: wandb.Video(frames, fps=20, format='mp4')})
                video_ep += 1

        logger.print('num_steps', num_steps)
        assert (
            calc_remaining_steps(num_steps) == 0
        ), f"The number of files does not match!!: {buffer_dir}\ndetected remaining step: {calc_remaining_steps(num_steps)}"

    else:
        logger.print(f"Buffer is filled: {buffer_dir}")

        from .adapt import verify_local_buffer
        logger.print('num_steps', num_steps)
        if not verify_local_buffer(buffer_dir, num_steps * action_repeat):
            num_files = len(list(buffer_dir.iterdir()))
            raise RuntimeError(
                f'Donwloaded buffer contains {num_files} files!! Aborting.'
            )

    return replay

def collect_orig_latent_buffer(
        agent, buffer_dir,
        env_name, buffer_size, nstep, discount, frame_stack, action_repeat, seed, time_limit,
        Adapt
):
    """Collect episodes in source (= original) domain.
    Encoded latents are stored to the buffer.
    """
    from .replay_buffer import Replay
    from .env_helpers import get_env
    from .agent import RandomShiftsAug

    buffer_dir = Path(buffer_dir)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    replay = Replay(
        buffer_dir, buffer_size=Adapt.latent_buffer_size,
        batch_size=Adapt.batch_size, num_workers=1,  # NOTE: Batch size does not matter at all
        nstep=nstep, discount=discount
    )
    clean_buffer_seed = seed + 1000
    logger.print('action_repeat (collect_orig_latent_buf)', action_repeat)
    clean_env = get_env(env_name, frame_stack, action_repeat, clean_buffer_seed, save_info_wrapper=True)
    collect_trajectories(
        agent=agent, env=clean_env, buffer_dir=buffer_dir, num_steps=Adapt.latent_buffer_size // action_repeat,
        pretrain_last_step=None, replay=replay, store_states=False,
        augment=RandomShiftsAug(pad=4) if Adapt.augment else None, action_repeat=action_repeat,
        video_key='original_latent_trajectory', encode_obs=True, random_policy=True,
        time_limit=time_limit
    )


def collect_targ_obs_buffer(
        buffer_dir,
        env_name, buffer_size, nstep, discount, frame_stack, action_repeat, seed, time_limit, Adapt
):
    """Collect episodes in target domain.
    Observations are stored rather than latents.
    """
    from .replay_buffer import Replay
    from .env_helpers import get_env

    buffer_dir = Path(buffer_dir)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    replay = Replay(
        buffer_dir, buffer_size=buffer_size,
        batch_size=Adapt.batch_size, num_workers=1,  # NOTE: Batch size does not matter at all
        nstep=nstep, discount=discount
    )
    logger.print('env_name', env_name)
    distr_env = get_env(env_name, frame_stack, action_repeat, seed,
                        distraction_config=Adapt.distraction_types, rails_wrapper=False,
                        intensity=Adapt.distraction_intensity)
    collect_trajectories(
        agent=None, env=distr_env, buffer_dir=buffer_dir, num_steps=Adapt.latent_buffer_size // action_repeat,
        pretrain_last_step=None, replay=replay , store_states=False, augment=None, action_repeat=action_repeat,
        video_key='target_obs_trajectory', encode_obs=False, random_policy=True,
        time_limit=time_limit
    )
