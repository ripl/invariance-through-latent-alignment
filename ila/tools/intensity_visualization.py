#!/usr/bin/env python3
"""A simple demo that produces an image from the environment."""
from typing import List, Tuple
import numpy as np
CELL_SIZE = 256

# domains = [
#             ('ball_in_cup', 'catch'),
#             ('cartpole', 'swingup'),
#             ('cheetah', 'run'),
#             ('finger', 'spin'),
#             ('reacher', 'easy'),
#             ('walker', 'walk')
# ]
# intensities = np.linspace(0, 1, 5+1)
domains = [('walker', 'walk')]
intensities = [0.0, 1.0]

def memoize(f):
    memo = {}

    def wrapper(*args, **kwargs):
        key = (*args, *kwargs.keys(), *kwargs.values())
        if key not in memo:
            memo[key] = f(*args, **kwargs)
        return memo[key]

    return wrapper


def compile_images(intensities: List[float], domains: List[Tuple[str, str]]):
    import wandb
    from ila.invr_thru_inf.env_helpers import get_env
    get_env = memoize(get_env)

    # We're retrieving a static image, thus these don't matter
    frame_stack = 3
    action_repeat = 3

    # seeds = [(i + 1) * 100 for i in range(5)]
    seeds = [200]
    for trial in range(5):
        for seed in seeds:
            for domain, task in domains:
                compiled_images = []
                for i in range(200):
                    rows = []
                    for distraction in ['color', 'camera', 'background']:
                        imgs = []
                        for intensity in intensities:

                            if distraction == 'color':
                                intensity *= 0.5
                            if distraction == 'camera':
                                intensity *= 0.25
                            print(distraction, domain, task, intensity)

                            name = f'distracting_control:{domain.capitalize()}-{task}-intensity-v1'
                            env = get_env(name, frame_stack, action_repeat, seed=seed,
                                        distraction_config=(distraction,), intensity=intensity)
                            # env.reset()
                            if i == 0:
                                env.reset()
                            else:
                                env.step(env.action_space.sample())
                            imgs.append(env.unwrapped.env.physics.render(height=CELL_SIZE, width=CELL_SIZE, camera_id=0))

                        rows.append(np.hstack(imgs))
                    compiled_images.append(np.vstack(rows))
                video = np.array(compiled_images).transpose(0, 3, 1, 2)
                wandb.log({f'compiled_video/{domain}-{task}-{seed}': wandb.Video(video, fps=20, format='mp4')})


if __name__ == '__main__':
    import os
    from ila.helpers.tticslurm import prepare_launch
    import numpy as np
    import wandb

    # Set prefix
    # job_name = kwargs['job_name']
    from ila import RUN
    # RUN.prefix = f'{RUN.project}/{job_name}'

    prepare_launch()

    # Initialize wandb
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=RUN.project + '-' + 'intensity-vis',
        # Track hyperparameters and run metadata
        # config=kwargs,
        # resume=True,
        # id=wandb_runid
    )

    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')

    # Controlled exit before slurm kills the process
    intensities = list(np.linspace(0, 1, 5+1))
    domains = [('Walker', 'walk')]
    print(intensities, domains)
    compile_images(intensities, domains)
