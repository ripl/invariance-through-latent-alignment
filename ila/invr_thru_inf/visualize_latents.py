#!/usr/bin/env python3
import os
from os.path import join as pjoin
import torch
import pandas as pd
from iti.helpers.media import tile_stacked_images

import wandb
from iti.helpers import logger, mllogger
from . import utils
from copy import deepcopy

# NOTE:
# 0. Prepare a pair of random trajectories both clean & distr.
# 1. Sample a random traj without distraction
# 2. Encode all into the latent space (100dim) with pretrained encoder
# 3. Sample a random traj with distraction (<-- *Paired* trajectory)
# 4. Encode all into the latent space (100dim) with pretrained encoder


def get_paired_obs(env, rails_env, agent=None):
    # Just sample one episode!
    done = False
    sim_states = []
    actions = []
    rewards = []
    obs = env.reset()
    src_obs = [obs]
    while not done:
        if agent is None:
            action = env.action_space.sample()
        else:
            eval_agent = deepcopy(agent)
            with torch.no_grad(), utils.eval_mode(eval_agent):
                action = eval_agent.act(obs)

        obs, rew, done, info = env.step(action)
        src_obs.append(obs)
        sim_states.append(info['sim_state'])
        actions.append(action)
        rewards.append(rew)

    # Now replay it in distracting env
    rails_env.set_episode({'state': sim_states, 'action': actions, 'reward': rewards})
    done = False
    obs = rails_env.reset()
    targ_obs = [obs]
    while not done:
        obs, rew, done, info = rails_env.step([0])
        targ_obs.append(obs)
    return src_obs, targ_obs


def draw_projection(agent, adpt_agent, orig_observations, targ_observations, key):
    # TEMP: Dirty but let me just duplicate the entire for block.
    df = pd.DataFrame(columns=['image', 'label', *[f'dim{idx:03d}' for idx in range(100)]])
    for idx, (orig_obs, targ_obs) in enumerate(zip(orig_observations, targ_observations)):

        orig_img = wandb.Image(tile_stacked_images(orig_obs))
        targ_img = wandb.Image(tile_stacked_images(targ_obs))
        # adpt_img = wandb.Image(tile_stacked_images(adpt_obs))
        with torch.no_grad(), utils.eval_mode(agent), utils.eval_mode(adpt_agent):
            orig_latent = agent.encode(orig_obs, augment=Adapt.augment, to_numpy=True)
            targ_latent = agent.encode(targ_obs, augment=Adapt.augment, to_numpy=True)
            adpt_latent = adpt_agent.encode(targ_obs, augment=Adapt.augment, to_numpy=True)

        orig_entry = {'image': orig_img, 'label': 'a_source', **{f'dim{key:03d}': val for key, val in enumerate(orig_latent.flatten())}}
        targ_entry = {'image': targ_img, 'label': 'b_target', **{f'dim{key:03d}': val for key, val in enumerate(targ_latent.flatten())}}
        adpt_entry = {'image': targ_img, 'label': 'c_adapted', **{f'dim{key:03d}': val for key, val in enumerate(adpt_latent.flatten())}}
        df = df.append(pd.Series(orig_entry), ignore_index=True) # , name=f'orig_{idx:06d}'))
        df = df.append(pd.Series(targ_entry), ignore_index=True) # , name=f'targ_{idx:06d}'))
        df = df.append(pd.Series(adpt_entry), ignore_index=True) # , name=f'targ_{idx:06d}'))

    wandb.log({key: wandb.Table(dataframe=df)})


def main(**kwargs):
    from iti.dmc_gen.config import Args
    from .config import Adapt
    from .env_helpers import get_env
    from iti.dmc_gen.adapt import load_adpt_agent

    assert 'reacher' not in Args.env_name.lower(), 'Reacher cannot be used as it is somwhow impossible to correctly store the goal location'

    Args._update(kwargs)
    Adapt._update(kwargs)

    # Create environment
    orig_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed + 1000,
                       size=Args.image_size, save_info_wrapper=True)
    targ_rails_env = get_env(Args.eval_env_name, Args.frame_stack, Args.action_repeat, Args.seed,
                                distraction_config=Adapt.distraction_types, size=Args.image_size,
                                intensity=Adapt.distraction_intensity, rails_wrapper=True)

    agent = load_adpt_agent(Args, Adapt)

    orig_obs_random, targ_obs_random = get_paired_obs(orig_env, targ_rails_env)
    orig_obs_policy, targ_obs_policy = get_paired_obs(orig_env, targ_rails_env, agent=agent)
    orig_obs_both = orig_obs_random + orig_obs_policy
    targ_obs_both = targ_obs_random + targ_obs_policy

    # Load adapted agents NOTE: Hardcoded!!
    from iti.invr_thru_inf.adapt import load_snapshot
    if Adapt.encoder_from_scratch:
        adpt_prefix = Args.job_name.replace('latent_vis_encscr/', 'iti/from_scratch_encoder/')
    else:
        adpt_prefix = Args.job_name.replace('latent_vis/', 'iti/adapt/')
        adpt_prefix = Args.job_name.replace('latent_vis2/', 'iti/adapt/')

    adpt_agent = load_snapshot(Args.checkpoint_root, completed=True, custom_prefix=adpt_prefix)
    # Read wandb id for adaptation and store it to summary
    try:
        adpt_wandb_runid = mllogger.read_params('wandb_runid', path=pjoin(Args.checkpoint_root, adpt_prefix, 'parameters.pkl'))
        wandb.run.summary['adaptation_wandb_runid'] = adpt_wandb_runid
    except KeyError as e:
        print('KeyError', e)

    # Evaluate pretrained, zero-shot and adapted agent:
    from .evaluate import eval
    orig_eval_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed + 1000,
                       size=Args.image_size)
    targ_eval_env = get_env(Args.eval_env_name, Args.frame_stack, Args.action_repeat, Args.seed,
                            distraction_config=Adapt.distraction_types, size=Args.image_size,
                            intensity=Adapt.distraction_intensity)
    eval(orig_eval_env, agent, 30, action_repeat=Args.action_repeat, to_video=True, log_prefix='pretrained')
    eval(targ_eval_env, agent, 30, action_repeat=Args.action_repeat, to_video=True, log_prefix='zeroshot')
    eval(targ_eval_env, adpt_agent, 30, action_repeat=Args.action_repeat, to_video=True, log_prefix='adapted')



    draw_projection(agent, adpt_agent, orig_obs_random, targ_obs_random, key='random_latents')
    draw_projection(agent, adpt_agent, orig_obs_policy, targ_obs_policy, key='policy_latents')
    draw_projection(agent, adpt_agent, orig_obs_both, targ_obs_both, key='all_latents')

if __name__ == '__main__':
    # Parse arguments
    import argparse
    from iti.dmc_gen.config import Args
    from iti.invr_thru_inf.config import Adapt
    from iti.helpers.tticslurm import prepare_launch

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str,
                        help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    from params_proto.neo_hyper import Sweep
    sweep = Sweep(Args, Adapt).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    # Set prefix
    job_name = kwargs['job_name']
    from iti import RUN
    RUN.prefix = f'{RUN.project}/{job_name}'

    prepare_launch(Args.job_name)

    # Obtain / generate wandb_runid
    # params_path = pjoin(Args.checkpoint_root, __prefix__, 'parameters.pkl')
    # try:
    #     wandb_runid = mllogger.read_params("wandb_runid", path=params_path)
    # except Exception as e:
    #     print(e)
    #     wandb_runid = wandb.util.generate_id()
    #     mllogger.log_params(wandb_runid=wandb_runid, path=params_path)

    # logger.info('params_path', params_path)
    # logger.info('If wandb initialization fails with "conflict" error, deleting ^ forces to generate a new wandb id (but all intermediate progress will be lost.)')
    # logger.print('wandb_runid:', wandb_runid)

    # Initialize wandb
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=RUN.project + '-vis-latents2',
        # Track hyperparameters and run metadata
        config=kwargs,
        # resume=True,
        # id=wandb_runid
    )

    wandb.run.summary['completed'] = wandb.run.summary.get('completed', False)

    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')

    # Controlled exit before slurm kills the process
    try:
        main(**kwargs)
        wandb.run.summary['completed'] = True
        wandb.finish()
    except TimeoutError as e:
        wandb.mark_preempting()
        raise e
