"""A simple demo that produces an image from the environment."""
import os
from ml_logger import logger
import numpy as np

import gym
import time

class Time:
    accm_saveimg = 0
    accm_step = 0
    accm_reset = 0

from contextlib import contextmanager

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

@contextmanager
def mllogger_root_to(value=''):
    org_ml_logger_root = os.environ.get('ML_LOGGER_ROOT', "")
    os.environ['ML_LOGGER_ROOT'] = value
    try:
        yield
    finally:
        os.environ['ML_LOGGER_ROOT'] = org_ml_logger_root


def memoize(f):
    memo = {}

    def wrapper(*args, **kwargs):
        key = (*args, *kwargs.keys(), *kwargs.values())
        if key not in memo:
            memo[key] = f(*args, **kwargs)
        return memo[key]

    return wrapper


def compile_images(outdir='reprod_figures/intensities', intensities=intensities, domains=domains):
    from ila.invr_thru_inf.env_helpers import get_env
    from .config import Args, Agent
    import distracting_control

    get_env = memoize(get_env)

    seeds = [(i + 1) * 100 for i in range(5)]
    for seed in seeds:
        for domain, task in domains:
            for i in range(50):
                rows = []
                for distraction in ['color', 'camera', 'background']:
                    prefix = f"{outdir}/{domain}-{task}"
                    imgs = []
                    for intensity in intensities:

                        if distraction == 'color':
                            intensity *= 0.5
                        if distraction == 'camera':
                            intensity *= 0.25
                        print(distraction, prefix, intensity)

                        name = f'distracting_control:{domain.capitalize()}-{task}-intensity-v1'
                        env = get_env(name, Args.frame_stack, Args.action_repeat, seed=seed,
                                      distraction_config=(distraction,), intensity=intensity)
                        # env = gym.make(
                        #     f'distracting_control:{domain.capitalize()}-{task}-intensity-v1',
                        #     from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True,
                        #     intensity=intensity, distraction_types=(distraction,), sample_from_edge=True
                        # )
                        if i == 0:
                            env.reset()
                        else:
                            env.step(env.action_space.sample())
                        imgs.append(env.unwrapped.env.render(height=128, width=128))

                    rows.append(np.hstack(imgs))
                logger.save_image(np.vstack(rows), prefix + f"-intensities-{i:03d}/{seed}.png")
                for j, img in enumerate(imgs):
                    logger.save_image(img, prefix + f"-separate-{i:03d}/{seed}/{j:03d}.png")


def record_video(outdir='reprod_figures/intensities/video'):
    domain, task = 'cheetah', 'run'
    intensity = 0.3
    for distraction in ['color', 'camera']:
        for distr_seed in [val * 100 for val in range(5)]:
            prefix = f"{outdir}/{distraction}/{domain}-{task}-intensity{intensity}-seed{distr_seed}"
            n_trials = 5
            episode_length = 10
            for j in range(n_trials):
                print(prefix, j)

                env = gym.make(
                    f'distracting_control:{domain.capitalize()}-{task}-intensity-v1',
                    from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True,
                    intensity=intensity, distraction_types=(distraction, ), sample_from_edge=True,
                    distraction_seed=distr_seed
                )
                obs = env.reset()

                done = False
                frames = []
                for k in range(episode_length):
                    frames.append(env.unwrapped.env.render(height=256, width=256))
                    act = env.action_space.sample()
                    obs, reward, done, info = env.step(act)
                    if done:
                        break

                logger.save_video(frames, prefix + f'ep{j}.mp4')


if __name__ == '__main__':
    print("---------- genrating images ----------")
    compile_images()
    # print("---------- genrating videos ----------")
    # record_video()
