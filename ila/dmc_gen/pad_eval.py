import torch
from os.path import join as pJoin
import numpy as np

from . import utils
from .import augmentations
from copy import deepcopy
from tqdm import tqdm


def load_final_checkpoint(path):
    from ml_logger import logger
    print('target', path)
    agent = mllogger.load_torch(path=path)
    return agent


def evaluate(env, agent, video, num_episodes, adapt=False, mean=True):
    from ml_logger import logger
    from .config import Args

    episode_rewards = []
    for i in tqdm(range(num_episodes)):
        if adapt:
            ep_agent = deepcopy(agent)
            ep_agent.init_pad_optimizer()
        else:
            ep_agent = agent
        obs = env.reset()
        # video.init(enabled=True)
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(ep_agent):
                action = ep_agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            # video.record(env, eval_mode)
            episode_reward += reward
            if adapt:
                ep_agent.update_inverse_dynamics(
                    *augmentations.prepare_pad_batch(obs, next_obs, action)
                )
            obs = next_obs

        # video.save(f"eval_{eval_mode}_{i}.mp4")
        episode_rewards.append(episode_reward)

        if adapt:
            logger.log({
                'online_episode_reward': episode_reward,
                'online_step': i,
                'online_frame': i * Args.episode_length,
                'wall_time': mllogger.since('start')
            }, flush=True)

    return np.mean(episode_rewards) if mean else episode_rewards


def main(**kwargs):
    from ml_logger import logger, ML_Logger
    from .config import Args
    from iti.invr_thru_inf.config import Adapt
    from iti.invr_thru_inf.env_helpers import get_env
    from iti.helpers.tticslurm import set_egl_id
    # Set seed
    set_egl_id()
    utils.set_seed_everywhere(kwargs['seed'])

    Args._update(kwargs)

    assert __prefix__, "you will overwrite the entire instrument server"
    if mllogger.read_params('job.completionTime', default=None):
        logger.print("The job seems to have been already completed!!!")
        return

    logger.start('start', 'episode', 'run')

    Adapt._update(kwargs)
    Args._update(kwargs)
    mllogger.log_params(Args=vars(Args))
    logger.log_text("""
        charts:
        - yKey: online_episode_reward
          xKey: online_step
        - yKey: episode_reward
          xKey: step
        """, filename=".charts.yml", dedent=True, overwrite=True)

    # Initialize environments
    clean_eval_env = get_env(
        Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed,
        distraction_config=Adapt.distraction_types, rails_wrapper=False,
        size=84 if Args.algorithm == 'sac' else 100)

    distr_eval_env = get_env(
        Args.eval_env_name, Args.frame_stack, Args.action_repeat, Args.seed,
        distraction_config=Adapt.distraction_types, intensity=Adapt.distraction_intensity,
        size=84 if Args.algorithm == 'sac' else 100
    )
    video = None

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    cropped_obs_shape = (
        3 * Args.frame_stack,
        Args.image_crop_size,
        Args.image_crop_size,
    )
    print("Observations:", clean_eval_env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)

    # Old location (s3)
    path = pJoin(Args.checkpoint_root, Adapt.snapshot_prefix, 'snapshot.pt')

    if not mllogger.glob(path):
        logger.print('glob failed on', path)
        s3_prefix = 's3://ge-data-improbable/drqv2_invariance'
        path = pJoin(s3_prefix, Adapt.snapshot_prefix, 'snapshot.pt')
        path = path.replace('/100', '-v1/100')  # TEMP
        path = path.replace('/200', '-v1/200')  # TEMP
        path = path.replace('/300', '-v1/300')  # TEMP
        path = path.replace('/400', '-v1/400')  # TEMP
        path = path.replace('/500', '-v1/500')  # TEMP
        path = path.replace('/run/', '/train/')  # TEMP
        logger.print('trying s3 path', path)

    agent = load_final_checkpoint(path)

    src_logger = ML_Logger(
        prefix=Adapt.snapshot_prefix
    )
    logger.print('src_logger', src_logger)
    keep_args = ['eval_env', 'seed', 'tmp_dir', 'checkpoint_root', 'time_limit']
    Args._update({key: val for key, val in src_mllogger.read_params("Args").items() if key not in keep_args})
    agent.train(False)

    results = {}

    reward = evaluate(clean_eval_env, agent, video, Args.eval_episodes, mean=False)
    logger.print('clean_eval', reward)
    reward = np.mean(reward)
    results['clean_eval'] = reward
    logger.print('clean_eval (mean)', reward)

    reward = evaluate(distr_eval_env, agent, video, Args.eval_episodes, mean=False)
    logger.print('before_adapt_reward', reward)
    reward = np.mean(reward)
    results['before_adapt_reward'] = reward
    logger.print('before_adapt_reward (mean)', reward)

    assert Args.algorithm == 'pad'
    adapt_reward = evaluate(
        distr_eval_env, agent, video, Args.eval_episodes, adapt=True
    )
    logger.log(episode_reward=adapt_reward, step=1, wall_time=mllogger.since('start'), flush=True)
    logger.print('after_adapt_reward', adapt_reward)
    adapt_reward = np.mean(adapt_reward)
    results['after_adapt_reward'] = adapt_reward
    logger.print('after_adapt_reward (mean)', adapt_reward)

    logger.save_pkl(results, 'results.pkl')

if __name__ == "__main__":
    main()
