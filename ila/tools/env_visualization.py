#!/usr/bin/env python3
import os
from typing import List
import wandb
from ila.helpers import logger


def wandb_save_video(frames, key, fps=20):
    import numpy as np
    if frames[0].shape[2] == 3:
        # (height, width, channel) --> (channel, height, width)
        frames = np.asarray([np.asarray(f).transpose(2, 0, 1) for f in frames])
    wandb.log({f'video/{key}': wandb.Video(frames, fps=fps, format='mp4')})


def main(env_name, distractions: List[str]):
    from ila.invr_thru_inf.env_helpers import get_env
    from .test_env_helpers import get_random_frames

    framestack = 3
    action_repeat = 3

    # Initially, visualize original (non-distr) domain!
    env = get_env(env_name, framestack, action_repeat,
                  seed=0, distraction_config=tuple(), size=84)

    num_episodes = 5
    for _ in range(num_episodes):
        _, rend_frames = get_random_frames(env, ep_len=1000, size=256)
        wandb_save_video(rend_frames, key=f'render-original')

    # for seed in [50 * i for i in range(11)]:
    for seed in [200]:
        logger.info(f'running with seed {seed}')
        env = get_env(env_name, framestack, action_repeat, seed,
                    distraction_config=distractions,
                    size=84, fix_distraction=True)

        num_episodes = 5
        for _ in range(num_episodes):
            _, rend_frames = get_random_frames(env, ep_len=1000, size=256)
            wandb_save_video(rend_frames, key=f'render-{seed}')


if __name__ == '__main__':
    import argparse
    from ila.helpers.tticslurm import prepare_launch

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='distracting_control:Walker-walk-easy-v1', help="env name (e.g., distracting_control:Walker-walk-v1)")
    parser.add_argument("--distr", nargs='+', default=['background'], help="type of distractions (e.g., background color camera)")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    # from params_proto.neo_hyper import Sweep
    # sweep = Sweep(Args).load(args.sweep_file)
    # kwargs = list(sweep)[args.line_number]

    # Set prefix
    # job_name = kwargs['job_name']
    from ila import RUN
    # RUN.prefix = f'{RUN.project}/{job_name}'

    print(args)

    prepare_launch()

    # Initialize wandb
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=RUN.project + '-' + 'env-vis',
        # Track hyperparameters and run metadata
        # config=kwargs,
        # resume=True,
        # id=wandb_runid
    )

    wandb.run.summary['completed'] = wandb.run.summary.get('completed', False)

    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')

    # Controlled exit before slurm kills the process
    try:
        main(args.env, args.distr)
        wandb.run.summary['completed'] = True
        wandb.finish()
    except TimeoutError as e:
        wandb.mark_preempting()
        raise e
