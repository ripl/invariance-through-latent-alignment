#!/usr/bin/env python3

import os
from .config import Agent, Args, Adapt
from ila.invr_thru_inf.env_helpers import get_env
import numpy as np


def save_obs(fname, obs):
    from ml_logger import logger
    assert obs.shape[0] % 3 == 0 and obs.shape[0] <= 9
    imgs = [obs[i*3:(i+1)*3, :, :] for i in range(int(obs.shape[0] / 3))]
    imgs = np.concatenate(imgs, axis=2)  # Stack images w.r.t. width
    logger.save_image(imgs.transpose(1, 2, 0), fname)


def main():
    from ml_logger import logger
    n_trials = 3

    org_ml_logger_root = os.environ.get('ML_LOGGER_ROOT', "")
    os.environ['ML_LOGGER_ROOT'] = ''

    for domain, task, seed in [
        ('ball_in_cup', 'catch', 100),
        ('cartpole', 'swingup', 100),
        ('cheetah', 'run', 200),
        ('finger', 'spin', 200),
        ('reacher', 'easy', 300),
        ('walker', 'walk', 300)
    ]:
        envname = f'dmc-gen:{domain.capitalize()}-{task}-v1'
        env = get_env(
            envname, Args.frame_stack,
            Args.action_repeat, seed
        )
        prefix = f"dmc-gen-examples/{domain}-{task}"
        for i in range(n_trials):
            obs = env.reset()
            done = False
            counter = 0
            frames = []
            while not done:
                frames.append(env.render(height=256, width=256))
                print(i, counter)
                save_obs(os.path.join(prefix, f'ep{i}-{counter}.png'), obs)
                # logger.save_image(
                #     obs, os.path.join(prefix, f"ep{i}-{counter}.png")
                # )
                act = env.action_space.sample()
                obs, reward, done, info = env.step(act)
                counter += 1
                if counter > 60:
                    break

            logger.save_video(frames, os.path.join(prefix, f'ep{i}.mp4'))
    os.environ['ML_LOGGER_ROOT'] = org_ml_logger_root


if __name__ == '__main__':
    main()
