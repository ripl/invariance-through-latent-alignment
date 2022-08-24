#!/usr/bin/env python3
from os.path import join as pjoin
from pathlib import Path

import torch
from params_proto.neo_proto import PrefixProto

from .utils import to_torch

import wandb
from ila.helpers import mllogger, logger
from ila import RUN

num_transitions_per_file = 1000  # NOTE: hardcoded


class Progress(PrefixProto, cli=False):
    step = 0
    wall_time = 0
    frame = 0
    invm_pretrain_done = 0  # HACK: Don't use bool. it'll be passed to logger.log_metrics.


def save_snapshot(agent, checkpoint_root, completed=True):
    params_prefix = pjoin(checkpoint_root, RUN.prefix)
    fname = 'snapshot_last.pt' if not completed else 'snapshot.pt'
    target_path = pjoin(checkpoint_root, RUN.prefix, fname)
    logger.print('Saving agent at', target_path)
    mllogger.save_torch(agent, path=target_path)
    mllogger.log_params(Progress=vars(Progress), path=pjoin(params_prefix, "checkpoint.pkl"), silent=True)


def load_snapshot(checkpoint_root, completed=False, custom_prefix=None):
    prefix = custom_prefix if custom_prefix else RUN.prefix
    fname = 'snapshot_last.pt' if not completed else 'snapshot.pt'
    target_path = pjoin(checkpoint_root, prefix, fname)
    logger.print('Loading agent from', target_path)
    adapt_agent = mllogger.load_torch(path=target_path)
    return adapt_agent


def verify_local_buffer(buffer_dir, latent_buffer_size):
    import math
    if buffer_dir.is_dir():
        num_expected_files = math.ceil(latent_buffer_size / num_transitions_per_file)
        num_files = len(list(buffer_dir.iterdir()))
        if num_files == num_expected_files:
            return True
    return False


def pretrain_dynamics(Adapt, agent, distr_replay, latent_replay, device):
    from .evaluate import evaluate_invm, evaluate_fwdm

    for step in range(Adapt.num_invm_steps):
        Progress.wall_time = mllogger.since('start')
        batch = next(latent_replay.iterator)
        latent, action, reward, discount, next_latent = to_torch(batch, device)
        action = action.to(torch.float32)  # TEMP: force float action
        # agent.update_invm(latent, next_latent, action)
        agent.learn_dynamics(latent, next_latent, action)

        # TODO: Use wandb here!
        # if mllogger.every(Adapt.invm_log_freq, key='dynamics_log'):
        #     logger.log_metrics_summary(
        #         key_values={'step': step, 'wall_time': Progress.wall_time}.items(),
        #         default_stats='mean'
        #     )

        if mllogger.every(Adapt.invm_eval_freq, key='dynamics_adaptation', start_on=1):
            distr_loss, clean_loss = evaluate_invm(agent, latent_replay, distr_replay)
            wandb.log({'pret_eval/invm_distr': distr_loss, 'pret_eval/invm_clean': clean_loss}, commit=False)

            # distr_loss, clean_loss = evaluate_fwdm(agent, latent_replay, distr_replay)
            # wandb.log({'fwdm_distr': distr_loss, 'fwdm_clean': clean_loss}, commit=False)
            wandb.log({'pretrain_step': step})



def adapt_offline(Adapt,
                  checkpoint_root, action_repeat, time_limit, batch_size, device,
                  orig_eval_env, targ_eval_env, agent, latent_replay, distr_replay,
                  progress=Progress):
    """
    clean_env, distr_env --> only for evalutation purposes!
    """
    from .evaluate import evaluate

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    if (Adapt.invm_lr > 0 or Adapt.fwdm_lr > 0) and not progress.invm_pretrain_done:
        logger.print('Starting invm pretraining...')
        pretrain_dynamics(Adapt, agent, distr_replay, latent_replay, device)
        logger.print('invm pretraining has finished')

        progress.invm_pretrain_done = 1  # HACK: Don't use bool. it'll be passed to logger.log_metrics.

    for progress.step in range(progress.step, Adapt.num_adapt_steps + 1):
        progress.wall_time = mllogger.since('start')
        progress.frame = progress.step * batch_size

        if time_limit and mllogger.since('run') > time_limit:
            logger.print(f'local time_limit: {time_limit} (sec) has reached!')
            logger.print('Saving snapshot...\t{vars(progress)}')
            save_snapshot(agent, checkpoint_root, completed=False)
            logger.print('Saving snapshot: Done!')
            raise TimeoutError

        # Evaluate current agent
        if mllogger.every(Adapt.adapt_eval_every_steps, key="adapt_eval", start_on=1):
            evaluate(
                agent, orig_eval_env, targ_eval_env, latent_replay, distr_replay,
                action_repeat,
                num_eval_episodes=Adapt.num_eval_episodes,
                progress=progress,
                eval_invm=(progress.step * action_repeat > Adapt.num_adapt_seed_frames)
            )

        if Adapt.gan_lr > 0:
            # Adversarial training
            for _ in range(Adapt.num_discr_updates):
                batch = next(distr_replay.iterator)
                obs, action, reward, discount, next_obs = to_torch(batch, device)
                action = action.to(torch.float32)  # TEMP: force float action

                distr_latent = agent.encode(obs, augment=Adapt.augment, to_numpy=False)
                agent.update_discriminator(latent_replay.iterator, distr_latent, improved_wgan=Adapt.improved_wgan)

            agent.update_encoder(obs)
        else:
            batch = next(distr_replay.iterator)
            obs, action, reward, discount, next_obs = to_torch(batch, device)
            action = action.to(torch.float32)  # TEMP: force float action

        # Update inverse dynamics
        if (Adapt.invm_lr > 0 or Adapt.fwdm_lr > 0) and progress.step >= Adapt.invm_kickin_step:
            agent.update_encoder_with_dynamics(obs, action, next_obs, update_model=Adapt.adapt_dyn_model)


    logger.print('===== Training completed!! =====')
    logger.print('Saving snapshot...')
    save_snapshot(agent, checkpoint_root, completed=True)
    logger.print('Done!')


def prepare_buffers(checkpoint_root, tmp_dir, train_env, eval_env, action_repeat, seed, Adapt):
    import math
    from .utils import get_buffer_prefix

    # This is used when the *corresponding envs to source buffer and target buffer are different* !
    targ_env = train_env if not Adapt.targ_buffer_name else Adapt.targ_buffer_name

    orig_buffer_seed = seed + 1000

    orig_buffer_prefix = get_buffer_prefix(
        train_env, action_repeat, seed, eval_env=None, latent_buffer=True,
        snapshot_prefix=Adapt.snapshot_prefix, distraction_types=Adapt.distraction_types,
        augment=Adapt.augment,
        distraction_intensity=Adapt.distraction_intensity
    )
    orig_buffer = Path(tmp_dir) / 'data-collection' / orig_buffer_prefix / 'workdir' / 'orig_latent_buffer'

    targ_buffer_prefix = get_buffer_prefix(
        targ_env, action_repeat, seed, eval_env=eval_env, latent_buffer=False,
        snapshot_prefix=Adapt.snapshot_prefix, distraction_types=Adapt.distraction_types,
        distraction_intensity=Adapt.distraction_intensity
    )
    # targ_buffer_prefix = Path(str(targ_buffer_prefix).replace('/nonaug/', '/aug/'))
    # NOTE: target buffer always contains "non-augmented" observations (because collect_targ_obs_buffer forces augment=None).
    # This deals with the case that Adapt.augment was set to True during targ buffer collection.

    targ_buffer = Path(tmp_dir) / 'data-collection' / targ_buffer_prefix / 'workdir' / "targ_obs_buffer"


    logger.print('org_buffer_dir', orig_buffer)
    logger.print('targ_buffer', targ_buffer)

    # Offline training
    assert Adapt.adapt_offline, "online adaptation is not supported right now"
    assert Adapt.download_buffer

    remote_orig_buffer = pjoin(checkpoint_root, orig_buffer_prefix, 'orig_latent_buffer.tar')
    remote_targ_buffer = pjoin(checkpoint_root, targ_buffer_prefix, 'targ_obs_buffer.tar')

    # NOTE: target buffer always contains "non-augmented" observations.  <-- I guess that's wrong.. ?
    # remote_targ_buffer = str(remote_targ_buffer).replace('/nonaug/', '/aug/')

    # logger.print(f'mllogger.glob_gs {remote_orig_buffer}', mllogger.glob_gs(remote_orig_buffer[5:]))
    assert mllogger.glob(remote_orig_buffer), f'orig latent buffer is not found at {remote_orig_buffer}'
    assert mllogger.glob(remote_targ_buffer), f'targ obs buffer is not found at {remote_targ_buffer}'

    logger.print('org_latent_buffer', orig_buffer)
    logger.print('verify_local_buff', verify_local_buffer(orig_buffer, Adapt.latent_buffer_size))
    logger.print('targ_obs_buffer', targ_buffer)
    logger.print('verify_local_buff', verify_local_buffer(targ_buffer, Adapt.latent_buffer_size))
    if not verify_local_buffer(orig_buffer, Adapt.latent_buffer_size):
        logger.print('downloading & unpacking buffer archive from', remote_orig_buffer)
        # mllogger.download_dir(remote_orig_buffer, to=orig_buffer.parents[0], unpack='tar')
        mllogger.download_dir(remote_orig_buffer, to=orig_buffer, unpack='tar')

    if not verify_local_buffer(targ_buffer, Adapt.latent_buffer_size):
        logger.print('downloading & unpacking buffer archive from', remote_targ_buffer)
        mllogger.download_dir(remote_targ_buffer, to=targ_buffer, unpack='tar')

    return orig_buffer, targ_buffer
