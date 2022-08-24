import os
from warnings import simplefilter  # noqa
from os.path import join as pjoin
from pathlib import Path

import gym
import numpy as np
from params_proto.neo_proto import PrefixProto

from . import utils
from .algorithms.factory import make_agent
from ila.invr_thru_inf.env_helpers import get_env
# from tools.notifier.slack_sender import slack_sender

import wandb
from ila.helpers import mllogger, logger

simplefilter(action='ignore', category=DeprecationWarning)
gym.logger.set_level(40)


class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0
    wall_time = 0
    frame = 0


def save_final_checkpoint(agent):
    # NOTE:
    # saving weights --> on slurm shared dir
    # saving hyper params -->  W & B
    # TODO: needs to specify a project prefix and share it across codebase
    from .config import Args
    run_prefix = pjoin(Args.checkpoint_root, __prefix__)

    snapshot_target = pjoin(Args.checkpoint_root, __prefix__, 'snapshot.pt')
    mllogger.save_torch(agent, path=snapshot_target)

    # Save actor.state_dict()
    snapshot_target = pjoin(Args.checkpoint_root, __prefix__, 'actor_state.pt')
    mllogger.save_torch(agent.actor.state_dict(), path=snapshot_target)

    mllogger.log_params(Progress=vars(Progress), path=pjoin(run_prefix, "progress.pkl"), silent=True)
    wandb.run.summary["progress"] = Progress  # NOTE: lets see if it works!
    wandb.run.summary["completed"] = True
    # mllogger.job_completed()


def save_checkpoint(agent, replay):
    from .config import Args
    import cloudpickle
    run_prefix = pjoin(Args.checkpoint_root, __prefix__)

    logger.print('saving & uploading snapshot...')
    replay_path = pjoin(run_prefix, 'replay.pt')
    snapshot_target = pjoin(run_prefix, 'snapshot_in_progress.pt')
    mllogger.save_torch(agent, path=snapshot_target)

    # mllogger.duplicate("metrics.pkl", "metrics_latest.pkl")
    logger.print('saving buffer to', replay_path)

    # NOTE: It seems tempfile cannot handle this 20GB+ file.
    mllogger.start('upload_replay_buffer')
    if replay_path.startswith('file://'):
        replay_path = replay_path[7:]
        Path(replay_path).resolve().parents[0].mkdir(parents=True, exist_ok=True)
        with open(replay_path, 'wb') as f:
            cloudpickle.dump(replay, f)
    elif replay_path.startswith('s3://') or replay_path.startswith('gs://'):
        logger.print('uploading buffer to', replay_path)
        tmp_path = Path(Args.tmp_dir) / __prefix__ / 'replay.pt'
        tmp_path.parents[0].mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'wb') as f:
            cloudpickle.dump(replay, f)

        if replay_path.startswith('s3://'):
            mllogger.upload_s3(str(tmp_path), path=replay_path[5:])
        else:
            mllogger.upload_gs(str(tmp_path), path=replay_path[5:])
    else:
        ValueError('replay_path must start with s3://, gs:// or file://. Not', replay_path)

    elapsed = mllogger.since('upload_replay_buffer')
    logger.print(f'Uploading replay buffer took {elapsed} seconds')

    # mllogger.log_params(Progress=vars(Progress), path="progress.pkl", silent=True)
    # mllogger.log_params(Progress=vars(Progress), path=pjoin(run_prefix, "progress.pkl"))
    wandb.run.summary["progress"] = Progress  # NOTE: lets see if it works!


def load_checkpoint():
    from .config import Args
    import torch

    # TODO: check if both checkpoint & replay buffer exist
    run_prefix = pjoin(Args.checkpoint_root, __prefix__)
    snapshot_path = pjoin(run_prefix, 'snapshot_in_progress.pt')
    replay_path = pjoin(run_prefix, 'replay.pt')
    progress_path = pjoin(run_prefix, 'progress.pkl')
    logger.print('snapshot_path', snapshot_path, mllogger.glob(snapshot_path))
    logger.print('replay_path', replay_path, mllogger.glob(replay_path))
    logger.print('progress.pkl', mllogger.glob(progress_path))
    logger.print('metrics_latest.pkl', mllogger.glob('metrics_latest.pkl'))

    # TODO: The pkl files needs to be fetched from W & B.
    # assert mllogger.glob(snapshot_path) and mllogger.glob(replay_path) and mllogger.glob('progress.pkl') and mllogger.glob('metrics_latest.pkl')
    assert mllogger.glob(snapshot_path) and mllogger.glob(replay_path) and mllogger.glob(progress_path)

    logger.print('loading agent from', snapshot_path)
    try:
        agent = mllogger.load_torch(path=snapshot_path)
        logger.print('load torch succeeds?', snapshot_path, agent)
    except RuntimeError as e:
        logger.print('captured exception', e)
        logger.print('unpicking failed. adding this run to a to-be-removed list')
        import os
        from .config import Args
        with open(os.path.expandvars('${HOME}/errored-runs.txt'), 'a') as f:
            f.write(pjoin(Args.checkpoint_root[7:], __prefix__) + '\n')
        raise e


    # Load replay buffer
    logger.print('loading from checkpoint (replay)', replay_path)

    # NOTE: It seems tempfile cannot handle this 20GB+ file.
    mllogger.start('download_replay_buffer')
    import pickle
    try:
        if replay_path.startswith('file://'):
            import cloudpickle
            with open(replay_path[7:], 'rb') as f:
                replay = cloudpickle.load(f)
        elif replay_path.startswith('s3://') or replay_path.startswith('gs://'):
            import cloudpickle
            tmp_path = Path(Args.tmp_dir) / __prefix__ / 'replay.pt'
            tmp_path.parents[0].mkdir(parents=True, exist_ok=True)

            if replay_path.startswith('s3://'):
                mllogger.download_s3(path=replay_path[5:], to=str(tmp_path))
            else:
                mllogger.download_gs(path=replay_path[5:], to=str(tmp_path))
            with open(tmp_path, 'rb') as f:
                replay = cloudpickle.load(f)
        else:
            ValueError('replay_path must start with s3://, gs:// or file://. Not', replay_path)
    except (pickle.UnpicklingError, EOFError) as e:
        logger.print('captured exception', e)
        logger.print('unpicking failed. adding this run to a to-be-removed list')
        import os
        from .config import Args
        with open(os.path.expandvars('${HOME}/errored-runs.txt'), 'a') as f:
            f.write(pjoin(Args.checkpoint_root[7:], __prefix__) + '\n')
        raise e


    elapsed = mllogger.since('download_replay_buffer')
    logger.print(f'Download completed. It took {elapsed} seconds')

    # mllogger.duplicate("metrics_latest.pkl", to="metrics.pkl")
    logger.print('done')

    # We can do it locally --> Does ml-mllogger support reading local pkl file??
    params = mllogger.read_params(path=progress_path)
    return agent, replay, params


def load_agent(checkpoint_root, prefix=None):
    from .config import Args
    import torch
    from ila import RUN
    prefix = RUN.prefix if prefix is None else prefix

    # TODO: check if both checkpoint & replay buffer exist
    run_prefix = pjoin(Args.checkpoint_root, prefix)
    snapshot_path = pjoin(run_prefix, 'snapshot.pt')
    logger.print('snapshot_path', snapshot_path, mllogger.glob(snapshot_path))
    assert mllogger.glob(snapshot_path)

    logger.print('loading agent from', snapshot_path)
    agent = mllogger.load_torch(path=snapshot_path)
    logger.print('load torch succeeds?', snapshot_path, agent)
    return agent

def evaluate(env, agent, num_episodes, video_path=None, train_step=None):
    episode_rewards = []
    frames = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        size = 256
        while not done:
            if i == 0 and video_path:
                frames.append(env.render(height=size, width=size))
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        if frames:
            frames.append(env.render(height=size, width=size))
            # NOTE: frames[0]: <PIL.Image.Image mode=RGB size=256x256 >
            # np.asarray(frames[0]).shape ==> (256, 256, 3)
            # mllogger.save_video(frames, video_path)
            # (height, width, channel) --> (channel, height, width)
            import time
            now = time.time()
            video = np.asarray([np.asarray(f).transpose(2, 0, 1) for f in frames])
            elapsed = time.time() - now
            logger.print(f'transpose and numpize the frames took {elapsed * 1000:.1f} ms')
            wandb.log({'eval': {"video": wandb.Video(video, fps=20, format="mp4")}, 'step': train_step})
            frames = []  # Hopefully this releases some memory
        episode_rewards.append(episode_reward)
    mean_r = np.mean(episode_rewards)
    wandb.log({'eval': {'episode_reward': wandb.Histogram(episode_rewards), 'episode_reward_mean': mean_r}, 'step': train_step})
    return mean_r


def train(train_env, eval_env, agent, replay_buffer, progress: Progress):
    from .config import Args
    done, episode_step, episode_reward = True, 0, 0

    for progress.step in range(progress.step, Args.train_steps + 1):
        progress.wall_time = mllogger.since('start')
        progress.frame = progress.step * Args.action_repeat

        if done:
            dt_episode = mllogger.split('episode')
            # mllogger.store_metrics({'train/episode_reward': episode_reward,
            #                       'dt_episode': dt_episode})
            # mllogger.log(step=progress.step, episode=progress.episode, wall_time=progress.wall_time, flush=True)
            wandb.log({"train": {"episode_reward": episode_reward, "dt_episode": dt_episode}}, commit=False)
            wandb.log({"step": progress.step, "episode": progress.episode, "wall_time": progress.wall_time})
            if episode_step:
                if progress.step % Args.log_freq == 0:
                    # mllogger.log_metrics_summary(key_values={'step': progress.step, 'episode': progress.episode})
                    wandb.log({'step': progress.step, 'episode': progress.episode})

                if progress.step % Args.eval_freq == 0:
                    logger.print('running evaluation...')
                    try:
                        with mllogger.Prefix(metrics="eval"):
                            video_path = f"videos/eval_agent_{progress.step:08d}.mp4"
                            mean_r = evaluate(eval_env, agent, Args.eval_episodes, video_path=video_path, train_step=progress.step)  # TEMP: ffmpeg  errors!
                            # mllogger.log({'episode_reward': mean_r})
                        # mllogger.log(step=progress.step, episode=progress.episode, flush=True)
                        wandb.log({'step': progress.step, 'episode': progress.episode})

                    except (OSError, RuntimeError) as e:
                        logger.print('Error captured:', e)
                        logger.print('Saving snapshot before exiting...')
                        save_checkpoint(agent, replay_buffer)
                        logger.print('done')
                        raise e
                    logger.print('running evaluation... done')

                if progress.step % Args.checkpoint_freq == 0:
                    save_checkpoint(agent, replay_buffer)

                if Args.time_limit and mllogger.since('run') > Args.time_limit:
                    logger.print(f'time limit {Args.time_limit} is reached. Saving snapshot...')
                    save_checkpoint(agent, replay_buffer)
                    raise TimeoutError

            obs, done, episode_reward, episode_step = train_env.reset(), False, 0, 0
            progress.episode += 1


        # Sample action for data collection
        if progress.step < Args.init_steps:
            action = train_env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if progress.step == Args.init_steps:
            logger.print(f'updating for {Args.init_steps} init steps')
            for _ in range(Args.init_steps):
                agent.update(replay_buffer, progress.step)
        elif progress.step > Args.init_steps:
            for _ in range(1):
                agent.update(replay_buffer, progress.step)

        # Take step
        next_obs, reward, done, _ = train_env.step(action)

        # always set done to False to make DMC infinite horizon. -- Ge
        replay_buffer.add(obs, action, reward, next_obs, False)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    # saving the final agent
    logger.print('Training is done!! Saving snapshot...')
    save_final_checkpoint(agent)
    logger.print('Saving snapshot done!')


# NOTE: This wrapper will do nothing unless $SLACK_WEBHOOK_URL is set
webhook_url = os.environ.get("SLACK_WEBHOOK_URL", None)


# @slack_sender(
#     webhook_url=webhook_url,
#     channel="rl-under-distraction-job-status",
#     progress=Progress,
#     ignore_exceptions=(TimeoutError,)
# )
def main(**kwargs):
    from ila import RUN
    from .config import Args
    run_prefix = pjoin(Args.checkpoint_root, __prefix__)

    Args._update(kwargs)

    # TODO: Read from W & B
    # Completion protection
    print('run_prefix', run_prefix)
    if wandb.run.summary.get('completed', False):
        logger.print("The job seems to have been already completed!!!")
        return
    # if logger.read_params('job.completionTime', path=pjoin(run_prefix, 'parameters.pkl'), default=None):
    #     logger.print("The job seems to have been already completed!!!")
    #     return

    mllogger.start('start', 'episode', 'run', 'step')

    # TODO: Read from W & B
    if RUN.resume and mllogger.glob(pjoin(run_prefix, 'progress.pkl')):

        # Update parameters
        keep_args = ['checkpoint_root', 'time_limit', 'checkpoint_freq', 'tmp_dir']
        Args._update({key: val for key, val in mllogger.read_params("Args", path=pjoin(run_prefix, 'parameters.pkl')).items() if key not in keep_args})

        agent, replay_buffer, progress_cache = load_checkpoint()
        Progress._update(progress_cache)
        mllogger.timer_cache['start'] = mllogger.timer_cache['start'] - Progress.wall_time
        logger.print(f'loaded from checkpoint at episode {Progress.episode}', color="green")

    else:
        logger.print("Start training from scratch.")

        # Apply kwargs parameters
        Args._update(kwargs)

        # logger.log_text("""
        #     charts:
        #     - yKey: train/episode_reward/mean
        #       xKey: step
        #     - yKey: eval/episode_reward
        #       xKey: step
        #     """, filename=".charts.yml", dedent=True, overwrite=True)

    mllogger.log_params(Args=vars(Args), path=pjoin(run_prefix, 'parameters.pkl'))

    utils.set_seed_everywhere(Args.seed)

    # NOTE: don't worry about the obs size. The size differs because it crops out the center.
    # shape of the input to the encoder stays the same.
    train_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed,
                        size=Args.image_size, distraction_config=Args.distraction_types)

    # NOTE: evaluation env includes all three distractions!!
    eval_env = get_env(Args.eval_env_name, Args.frame_stack, Args.action_repeat, Args.seed + 42,
                        size=Args.image_size, distraction_config=('background', 'color', 'camera'))

    if 'agent' not in locals():
        replay_buffer = utils.ReplayBuffer(obs_shape=train_env.observation_space.shape,
                                           action_shape=train_env.action_space.shape,
                                           capacity=Args.train_steps,
                                           batch_size=Args.batch_size)
        _cropped_obs_shape = (3 * Args.frame_stack, Args.image_crop_size, Args.image_crop_size)
        agent = make_agent(obs_shape=_cropped_obs_shape, action_shape=train_env.action_space.shape, args=Args)

    try:
        train(train_env, eval_env, agent, replay_buffer, progress=Progress)
        logger.print('Training is completed!')
        if Args.checkpoint_root:
            save_final_checkpoint(agent)
    except TimeoutError as e:
        if Args.checkpoint_root:
            save_checkpoint(agent, replay_buffer)
        raise e


if __name__ == '__main__':
    """
    Example:
    The following will launch an experiment with config in L1 of sweep.jsonl
    $ python -m learning_with_library.dmc_gen.train sweep.jsonl -l 1
    """

    import os
    import argparse
    from .config import Args
    from ila.helpers.tticslurm import prepare_launch
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str,
                        help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    from params_proto.neo_hyper import Sweep
    sweep = Sweep(Args).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    # Set prefix
    from ila import __prefix__, __project__
    __prefix__ = f'{__project__}/{Args.job_name}'

    prepare_launch(Args.job_name)

    logger.print('project name', __project__)
    logger.print('prefix', __prefix__)

    # Obtain / generate wandb_runid
    params_path = pjoin(Args.checkpoint_root, __prefix__, 'parameters.pkl')
    try:
        wandb_runid = mllogger.read_params("wandb_runid", path=params_path)
    except Exception as e:
        print(e)
        wandb_runid = wandb.util.generate_id()

    logger.print('wandb_runid:', wandb_runid)

    # Check if wandb_runid is stored or not.
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=__project__,
        # Track hyperparameters and run metadata
        config=kwargs,
        resume=True,
        id=wandb_runid
    )
    mllogger.log_params(wandb_runid=wandb_runid, path=params_path)

    wandb.run.summary['completed'] = wandb.run.summary.get('completed', False)

    # Controlled exit before slurm kills the process
    try:
        main(**kwargs)
        wandb.finish()
    except TimeoutError as e:
        wandb.mark_preempting()
        raise e
