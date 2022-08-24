#!/usr/bin/env python3

import torch
from .utils import to_torch
import wandb
from ila.helpers import logger

@torch.no_grad()
def evaluate_invm(agent, latent_replay, distr_replay):
    from ila.invr_thru_inf.dynamics import invm_loss

    n_samples = 512
    batch = next(distr_replay.iterator)
    batch_size = len(batch[0])
    # logger.print('batch size =================> ', batch_size)
    for _ in range(n_samples // batch_size):
        obs, action, reward, discount, next_obs = to_torch(batch, agent.device)
        latent = agent.encode(obs, to_numpy=False)
        next_latent = agent.encode(next_obs, to_numpy=False)

        assert agent.dynamics_model.head_key[0] == "invm"
        distr_loss = invm_loss(agent.dynamics_model.heads[0], latent, next_latent, action).item()
        # distr_loss = agent.get_invm_loss(latent, next_latent, action).item()
        batch = next(distr_replay.iterator)

    for _ in range(n_samples // batch_size):
        batch = next(latent_replay.iterator)
        latent, action, reward, discount, next_latent = to_torch(batch, agent.device)

        if len(latent.shape) == 3:
            # Sample one out of four in each instance
            # NOTE: latent: (batch_size, 4, 1024)
            batch_size, num_augments, _ = latent.shape
            latent = latent.mean(dim=1)
            next_latent = next_latent.mean(dim=1)

        assert agent.dynamics_model.head_key[0] == "invm"
        clean_loss = invm_loss(agent.dynamics_model.heads[0], latent, next_latent, action).item()
        # clean_loss = agent.get_invm_loss(latent, next_latent, action).item()

    return distr_loss, clean_loss


@torch.no_grad()
def evaluate_fwdm(agent, latent_replay, distr_replay):
    from ila.invr_thru_inf.dynamics import fwdm_loss

    n_samples = 512
    batch = next(distr_replay.iterator)
    batch_size = len(batch[0])
    logger.print('batch size =================> ', batch_size)
    for _ in range(n_samples // batch_size):
        obs, action, reward, discount, next_obs = to_torch(batch, agent.device)
        latent = agent.encode(obs, to_numpy=False)
        next_latent = agent.encode(next_obs, to_numpy=False)

        assert agent.dynamics_model.head_key[1] == "fwdm"
        distr_loss = fwdm_loss(agent.dynamics_model.heads[1], latent, next_latent, action).item()
        # distr_loss = agent.get_fwdm_loss(latent, next_latent, action).item()
        batch = next(distr_replay.iterator)

    for _ in range(n_samples // batch_size):
        batch = next(latent_replay.iterator)
        latent, action, reward, discount, next_latent = to_torch(batch, agent.device)

        if len(latent.shape) == 3:
            # Sample one out of four in each instance
            # NOTE: latent: (batch_size, 4, 1024)
            batch_size, num_augments, _ = latent.shape
            latent = latent.mean(dim=1)
            next_latent = next_latent.mean(dim=1)

        assert agent.dynamics_model.head_key[1] == "fwdm"
        clean_loss = fwdm_loss(agent.dynamics_model.heads[1], latent, next_latent, action).item()
        # clean_loss = agent.get_fwdm_loss(latent, next_latent, action).item()

    return distr_loss, clean_loss


def eval(env, agent,
         num_eval_episodes, action_repeat,
         to_video=None, stochastic_video=None, log_prefix=''):
    import time
    from copy import deepcopy
    from . import utils
    from statistics import mean

    step = 0
    ep_rewards = []
    for episode in range(num_eval_episodes):
        ep_reward = 0
        # NOTE: This takes ~0.01 second. Env rollout is much slower.
        eval_agent = deepcopy(agent)  # make a new copy
        obs = env.reset()
        frames = []
        done = False
        size = 100
        while not done:
            if episode == 0 and to_video:
                # todo: use gym.env.render('rgb_array') instead
                frames.append(env.render(height=size, width=size))

            with torch.no_grad(), utils.eval_mode(eval_agent):
                # todo: remove global_step, replace with random-on, passed-in.
                action = eval_agent.act(obs)
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            ep_reward += reward
            step += 1
        ep_rewards.append(ep_reward)

        if episode == 0 and to_video:
            import numpy as np
            # Append the last frame
            frames.append(env.render(height=size, width=size))
            # whc --> cwh
            frames = np.asarray([np.asarray(f).transpose(2, 0, 1) for f in frames])
            # frames = np.asarray([np.asarray(f) for f in frames])
            wandb.log({f'eval/{log_prefix}/video': wandb.Video(frames, fps=20, format='mp4')}, commit=False)
            # logger.save_video(frames, to_video)

    # Save video using stochastic agent (eval_mode is set to False)
    if stochastic_video:
        obs = env.reset()
        frames = []
        done = False
        while not done:
            frames.append(env.render(height=size, width=size))
            with torch.no_grad():
                action = eval_agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
        frames.append(env.render(height=size, width=size))
        logger.save_video(frames, stochastic_video)

    # logger.log(episode_reward=total_reward / episode, episode_length=step * action_repeat / episode)
    wandb.log({f'eval/{log_prefix}/ep_rewards': wandb.Histogram(ep_rewards), f'eval/{log_prefix}/ep_rewards_mean': mean(ep_rewards)}, commit=False)


def evaluate(agent, clean_eval_env, distr_eval_env,
             latent_replay, distr_replay, action_repeat, num_eval_episodes, progress, expl_agent=None, eval_invm=True):

    # if Progress.step * Args.action_repeat > Adapt.num_adapt_seed_frames:
    if eval_invm:
        # with logger.Prefix(metrics="eval"):
        #     # Compare latent mean and stddev
        #     logger.print('evaluating clean latent space')
        #     compare_latent_spaces(agent if expl_agent is None else expl_agent, latent_replay, distr_replay)

        # Evaluate inv_dynamics head on clean_env latent
        # NOTE: only use encoder
        logger.print('evaluating invm')
        distr_loss, clean_loss = evaluate_invm(agent if expl_agent is None else expl_agent, latent_replay, distr_replay)
        wandb.log({'eval/dynamics/invm/distr_loss': distr_loss, 'eval/dynamics/invm/clean_loss': clean_loss})

        logger.print('evaluating fwdm')
        distr_loss, clean_loss = evaluate_fwdm(agent if expl_agent is None else expl_agent, latent_replay, distr_replay)
        wandb.log({'eval/dynamics/fwdm/distr_loss': distr_loss, 'eval/dynamics/fwdm/clean_loss': clean_loss})

    if expl_agent:
        # Replace agent encoder weights
        org_enc_state = agent.encoder.state_dict()
        agent.encoder.load_state_dict(expl_agent.encoder.state_dict())

    # Evaluation on clean env
    logger.print('evaluating clean env')
    path = f'videos/clean/{progress.step:09d}_eval.mp4'
    stoch_path = None
    eval(clean_eval_env, agent, num_eval_episodes,
            action_repeat, to_video=path,
            stochastic_video=stoch_path, log_prefix='clean')

    # Evaluation on distr env
    logger.print('evaluating distr env')
    path = f'videos/distr/{progress.step:09d}_eval.mp4'
    stoch_path = None
    eval(distr_eval_env, agent, num_eval_episodes,
         action_repeat, to_video=path,
         stochastic_video=stoch_path, log_prefix='distr')

    if expl_agent:
        # with logger.Prefix(metrics="clean_eval_original"):
        #     logger.print('evaluating clean env with original policy')
        #     path = f'videos/clean-with-original-policy/{progress.step:09d}_eval.mp4'
        #     stoch_path = f'videos/clean-with-original-policy-stoch/{progress.step:09d}_eval.mp4'
        #     eval(clean_eval_env, agent, progress.step, to_video=path if Args.save_video else None,
        #          stochastic_video=stoch_path)
        with logger.Prefix(metrics="eval_expl"):
            logger.print('evaluating distr env with original policy')
            path = f'videos/distr-with-expl-policy/{progress.step:09d}_eval.mp4'
            stoch_path = None
            eval(distr_eval_env, expl_agent, to_video=path,
                 stochastic_video=stoch_path)

        # Put back the original encoder weights
        agent.encoder.load_state_dict(org_enc_state)

    # logger.log(**vars(progress))
    # logger.flush()
    wandb.log({'step': progress.step})
