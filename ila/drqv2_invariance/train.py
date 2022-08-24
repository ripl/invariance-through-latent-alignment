import os
from pathlib import Path
from warnings import filterwarnings  # noqa
from os.path import join as pJoin

import numpy as np
import torch
from params_proto.neo_proto import PrefixProto
from copy import deepcopy

from drqv2_invariance.replay_buffer import Replay
from . import utils
from .config import Args, Agent
from tools.notifier.slack_sender import slack_sender


class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0
    wall_time = 0
    frame = 0


def save_checkpoint(agent, replay_cache_dir, replay_upload_dir):
    from ml_logger import logger

    logger.print('saving checkpoint...')
    model_snapshot_path = pJoin(Args.checkpoint_root, __prefix__, 'snapshot.pt')
    mllogger.save_torch(agent, path=model_snapshot_path)
    if replay_upload_dir.startswith('file://'):
        logger.upload_dir(str(replay_cache_dir), pJoin(replay_upload_dir, 'replay'), archive=None)
    else:
        logger.upload_dir(str(replay_cache_dir), pJoin(replay_upload_dir, 'replay.tar'))
    logger.duplicate("metrics.pkl", "metrics_last.pkl")
    # Save progress at last to avoid any clutter
    mllogger.log_params(Progress=vars(Progress), path="checkpoint.pkl", silent=True)


def load_checkpoint(replay_checkpoint, replay_cache_dir):
    from ml_logger import logger

    logger.print('loading from checkpoint...')
    model_snapshot_path = pJoin(Args.checkpoint_root ,__prefix__, 'snapshot.pt')
    agent = mllogger.load_torch(path=model_snapshot_path)
    if replay_checkpoint.startswith('file://'):
        mllogger.download_dir(pJoin(replay_checkpoint, 'replay'), str(replay_cache_dir), unpack=None)
    else:
        mllogger.download_dir(pJoin(replay_checkpoint, 'replay.tar'), str(replay_cache_dir))
    logger.duplicate("metrics_last.pkl", to="metrics.pkl")

    return agent


def eval(env, agent, global_step, to_video=None, stochastic_video=None):
    from ml_logger import logger

    prev_step, step, total_reward = 0, 0, 0
    for episode in range(Args.num_eval_episodes):
        eval_agent = deepcopy(agent)  # make a new copy
        obs = env.reset()
        frames = []
        done = False
        size = 64
        while not done:
            if episode == 0 and to_video:
                # todo: use gym.env.render('rgb_array') instead
                frames.append(env.render(height=size, width=size))

            with torch.no_grad(), utils.eval_mode(eval_agent):
                # todo: remove global_step, replace with random-on, passed-in.
                action = eval_agent.act(obs, global_step, eval_mode=True)
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            step += 1

        if episode == 0 and to_video:
            # Append the last frame
            frames.append(env.render(height=size, width=size))
            logger.save_video(frames, to_video)

    # Save video using stochastic agent (eval_mode is set to False)
    if stochastic_video:
        obs = env.reset()
        frames = []
        done = False
        while not done:
            frames.append(env.render(height=size, width=size))
            with torch.no_grad():
                action = eval_agent.act(obs, global_step, eval_mode=False)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
        frames.append(env.render(height=size, width=size))
        logger.save_video(frames, stochastic_video)

    logger.log(episode_reward=total_reward / episode, episode_length=step * Args.action_repeat / episode)


def train(train_env, eval_env, agent, replay, progress: Progress):
    from ml_logger import logger

    init_transition = dict(
        obs=None, reward=0.0, done=False, discount=1.0,
        action=np.full(eval_env.action_space.shape, 0.0, dtype=eval_env.action_space.dtype)
    )

    episode_step, episode_reward = 0, 0
    obs = train_env.reset()
    transition = {**init_transition, 'obs': obs}
    replay.storage.add(**transition)
    done = False
    for progress.step in range(progress.step, Args.train_frames // Args.action_repeat + 1):
        progress.wall_time = mllogger.since('start')
        progress.frame = progress.step * Args.action_repeat

        if done:
            progress.episode += 1

            # log stats
            episode_frame = episode_step * Args.action_repeat
            logger.log(fps=episode_frame / logger.split('episode'),
                       episode_reward=episode_reward,
                       episode_length=episode_frame,
                       buffer_size=len(replay.storage))

            # reset env
            obs = train_env.reset()
            done = False
            transition = {**init_transition, 'obs': obs}
            replay.storage.add(**transition)
            # try to save snapshot
            if Args.time_limit and mllogger.since('run') > Args.time_limit:
                logger.print(f'local time_limit: {Args.time_limit} (sec) has reached!')
                raise TimeoutError

            episode_step, episode_reward = 0, 0

        # try to evaluate
        if mllogger.every(Args.eval_every_frames // Args.action_repeat, key="eval"):
            try:
                with logger.Prefix(metrics="eval"):
                    path = f'videos/{progress.step * Args.action_repeat:09d}_eval.mp4'
                    eval(eval_env, agent, progress.step, to_video=path if Args.save_video else None)
            except OSError as e:
                print("OSError captured", e)
                replay_checkpoint = pJoin(Args.checkpoint_root, __prefix__)
                replay_cache_dir = Path(Args.tmp_dir) / __prefix__ / 'replay'
                print("saving snapshot...")
                save_checkpoint(agent, replay_cache_dir, replay_checkpoint)
                raise e
            logger.log(**vars(progress))
            logger.flush()

        # sample action
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, progress.step, eval_mode=False)

        # try to update the agent
        if progress.step * Args.action_repeat > Args.num_seed_frames:
            if mllogger.every(Args.update_freq, key="update"):
                agent.update(replay.iterator, progress.step)  # checkpoint.step is used for scheduling
            # if mllogger.every(Args.log_freq, key="log"):
            #     logger.log_metrics_summary(vars(progress), default_stats='mean')

        # take env step
        obs, reward, done, info = train_env.step(action)
        episode_reward += reward

        # TODO: is it ok to always use discount = 1.0 ?
        # we should actually take a look at time_step.discount
        transition = dict(obs=obs, reward=reward, done=done, discount=1.0, action=action)
        replay.storage.add(**transition)
        episode_step += 1


# NOTE: This wrapper will do nothing unless $SLACK_WEBHOOK_URL is set
webhook_url = os.environ.get("SLACK_WEBHOOK_URL", None)


@slack_sender(
    webhook_url=webhook_url,
    channel="rl-under-distraction-job-status",
    progress=Progress,
    ignore_exceptions=(TimeoutError,)
)
def main(**kwargs):
    from ml_logger import logger, RUN
    from drqv2_invariance.drqv2_invar import DrQV2Agent
    from iti.invr_thru_inf.env_helpers import get_env
    import shutil

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)


    utils.set_egl_id()
    Args._update(kwargs)
    utils.set_seed_everywhere(Args.seed)

    assert __prefix__, "you will overwrite the experiment root"

    # Completion protection
    if mllogger.read_params('job.completionTime', default=None):
        logger.print("The job seems to have been already completed!!!")
        return

    logger.start('run')


    if RUN.resume and mllogger.glob('checkpoint.pkl'):
        replay_cache_dir = Path(Args.tmp_dir) / __prefix__ / 'replay'
        shutil.rmtree(replay_cache_dir, ignore_errors=True)
        replay_checkpoint = pJoin(Args.checkpoint_root, __prefix__)


        logger.start('episode')

        logger.print('before update Args', vars(Args))

        # Update parameters
        keep_args = ['seed', 'tmp_dir', 'checkpoint_root', 'Args.time_limit', 'device']
        Args._update({key: val for key, val in mllogger.read_params("Args").items() if key not in keep_args})

        agent = load_checkpoint(replay_checkpoint, replay_cache_dir)

        Progress._update(mllogger.read_params(path="checkpoint.pkl"))
        Agent._update(**mllogger.read_params('Agent'))
        mllogger.timer_cache['start'] = mllogger.timer_cache['episode'] - Progress.wall_time

        logger.print(f'loaded from checkpoint at episode {Progress.episode}', color="green")

    else:
        logger.print("Start training from scratch.")

        # Apply kwargs parameters
        Args._update(kwargs)
        Agent._update(kwargs)

        # todo: this needs to be fixed.
        logger.log_text("""
            keys:
            - Args.env_name
            - Args.seed
            charts:
            - yKey: eval/episode_reward
              xKey: frame
            - yKey: episode_reward
              xKey: frame
            - yKey: fps/mean
              xKey: wall_time
            - yKeys: ["batch_reward/mean", "critic_loss/mean"]
              xKey: frame
            """, ".charts.yml", overwrite=True, dedent=True)

        logger.start('start', 'episode')

    mllogger.log_params(Args=vars(Args), Agent=vars(Agent))

    utils.set_seed_everywhere(Args.seed)
    train_env = get_env(Args.train_env, Args.frame_stack, Args.action_repeat, Args.seed)
    eval_env = get_env(Args.eval_env, Args.frame_stack, Args.action_repeat, Args.seed)

    if 'agent' not in locals():
        agent = DrQV2Agent(obs_shape=train_env.observation_space.shape,
                           action_shape=train_env.action_space.shape,
                           device=Args.device, **vars(Agent))

    replay = Replay(
        buffer_dir=replay_cache_dir, buffer_size=Args.replay_buffer_size,
        batch_size=Args.batch_size, num_workers=Args.replay_buffer_num_workers,
        nstep=Args.nstep, discount=Args.discount
    )

    try:
        train(train_env, eval_env, agent, replay, progress=Progress)
        logger.print('Training is completed!')
        if Args.checkpoint_root:
            save_checkpoint(agent, replay_cache_dir, replay_checkpoint)
    except TimeoutError as e:
        if Args.checkpoint_root:
            save_checkpoint(agent, replay_cache_dir, replay_checkpoint)
        raise e


if __name__ == '__main__':
    main()
