#!/usr/bin/env python3
import os
from typing import List
import wandb
from iti.helpers import logger
from iti.invr_thru_inf.env_helpers import get_env
from .test_env_helpers import get_random_frames
from .env_visualization import wandb_save_video

def vis_episodes(env_name):
    framestack = 3
    action_repeat = 3
    env = get_env(env_name, framestack, action_repeat, seed=100, size=84)

    num_episodes = 60
    for _ in range(num_episodes):
        _, rend_frames = get_random_frames(env, ep_len=100, size=256)
        wandb_save_video(rend_frames, key=f'render')


def specific(env_name):
    from .test_env_helpers import get_random_frames
    from .env_visualization import wandb_save_video
    framestack = 3
    action_repeat = 3

    for _ in range(50):
        env = get_env(env_name, framestack, action_repeat, seed=0, size=84)

        env.viewer.cam.distance = 0.5
        env.viewer.cam.azimuth = 240
        env.viewer.cam.elevation = -15
        env.viewer.cam.lookat[0] += 0.1
        env.viewer.cam.lookat[1] = 1.347
        env.viewer.cam.lookat[2] = 0.77
        print(env.viewer.cam.lookat)
        # [1.50005132 1.347      0.77      ]

        env.reset()
        img = env.render()
        # img.shape == (500, 500, 3)
        from .test_env_helpers import get_random_frames

        wandb.log({'img': wandb.Image(img)})
    # wandb_save_video(rend_frames, key=f'render', fps=3)

def main(env_name):
    from iti.invr_thru_inf.env_helpers import get_env
    from .test_env_helpers import get_random_frames
    from .env_visualization import wandb_save_video

    framestack = 3
    action_repeat = 3

    rend_frames = []
    for dist in [0.5, 0.6, 0.7]:
        for elevation in [-15]:
            for azimuth in [235, 240, 245]:
                for lookatpy in [0.6]:
                    for lookatpz in [0.0, 0.1, 0.2, 0.3]:
                        for lookatpx in [0.0, 0.1, 0.2]:
                            env = get_env(env_name, framestack, action_repeat, seed=0, size=84)

                            env.viewer.cam.distance = dist        # 1.0
                            env.viewer.cam.azimuth = azimuth      # 240
                            env.viewer.cam.elevation = elevation  # -15
                            env.viewer.cam.lookat[0] += lookatpx
                            env.viewer.cam.lookat[1] += lookatpy  # 1.147
                            env.viewer.cam.lookat[2] += lookatpz  # 0.772

                            env.reset()
                            img = env.render()
                            # img.shape == (500, 500, 3)

                            import cv2
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1.
                            color = (0, 0, 0)
                            img = cv2.putText(img.copy(), f'dist: {dist}', (0, 30), font, fontScale, color)
                            img = cv2.putText(img, f'azimuth: {azimuth}', (0, 60), font, fontScale, color)
                            img = cv2.putText(img, f'elevation: {elevation}', (0, 90), font, fontScale, color)
                            img = cv2.putText(img, f'lookaty: {env.viewer.cam.lookat[1]}', (0, 120), font, fontScale, color)
                            img = cv2.putText(img, f'lookatz: {env.viewer.cam.lookat[2]}', (0, 150), font, fontScale, color)
                            rend_frames.append(img)
                            # _, rend_frames = get_random_frames(env, ep_len=100, size=256)
    wandb_save_video(rend_frames, key=f'render', fps=3)

if __name__ == '__main__':
    import argparse
    from iti.helpers.tticslurm import prepare_launch

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='fetch:Reach-floor-v0', help="env name (e.g., distracting_control:Walker-walk-v1)")
    args = parser.parse_args()

    # Set prefix
    # job_name = kwargs['job_name']
    from iti import RUN
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
        vis_episodes(args.env)
        # specific(args.env)
        # main(args.env)
        wandb.run.summary['completed'] = True
        wandb.finish()
    except TimeoutError as e:
        wandb.mark_preempting()
        raise e
