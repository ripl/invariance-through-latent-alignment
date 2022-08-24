#!/usr/bin/env python3
import os
from os.path import join as pjoin
from pathlib import Path

from ila.invr_thru_inf.collect_offline_data import (
    upload_and_cleanup, collect_orig_latent_buffer, collect_targ_obs_buffer
)

from ila import RUN
from ila.helpers import logger, mllogger
from ila.invr_thru_inf.env_helpers import parse_env_name

import wandb


def main(**kwargs):
    from .config import Args
    from ila.invr_thru_inf.config import Adapt, CollectData
    from ila.invr_thru_inf.utils import set_seed_everywhere, get_buffer_prefix

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    mllogger.start('run')

    Args._update(kwargs)
    Adapt._update(kwargs)

    set_seed_everywhere(kwargs['seed'])

    if Adapt.snapshot_prefix:

        params_path = pjoin(Adapt.snapshot_prefix, 'parameters.pkl')
        logger.print('reading from', params_path)
        # Check if the job is completed
        try:
            completion_time = mllogger.read_params('job.completionTime', path=params_path)
            logger.print(f'Pre-training was completed at {completion_time}.')
        except KeyError:
            logger.print(f'training for {Adapt.snapshot_prefix} has not been finished yet.')
            logger.print('job.completionTime was not found.')
            raise RuntimeError

        # Update parameters
        keep_args = ['eval_env_name', 'seed', 'tmp_dir', 'checkpoint_root', 'time_limit', 'device']
        Args._update({key: val for key, val in mllogger.read_params("Args", path=params_path).items() if key not in keep_args})
        logger.print('Args after update', vars(Args))
    params_prefix = pjoin(Args.checkpoint_root, RUN.prefix)
    mllogger.log_params(Args=vars(Args), Adapt=vars(Adapt), path=pjoin(params_prefix, 'parameters.pkl'))

    # Update config parameters based on kwargs
    CollectData._update(kwargs)

    # ===== Prepare the buffers =====

    # Currently only this configuration is suppoorted.
    assert Adapt.policy_on_clean_buffer == "random" and Adapt.policy_on_distr_buffer == "random"

    if CollectData.buffer_type == 'original':
        from .adapt import load_adpt_agent
        buffer_prefix = get_buffer_prefix(
            Args.env_name, Args.action_repeat, Args.seed, Adapt.snapshot_prefix,
            augment=Adapt.augment, latent_buffer=True
        )
        orig_buffer_dir = Path(Args.tmp_dir) / 'data-collection' / buffer_prefix / 'workdir' / 'orig_latent_buffer'
        logger.print('orig_buffer_dir', orig_buffer_dir)
        agent = load_adpt_agent(Args, Adapt)
        remote_orig_buffer = pjoin(Args.checkpoint_root, buffer_prefix, 'orig_latent_buffer.tar')
        logger.print(remote_orig_buffer)

        if mllogger.glob(remote_orig_buffer):
            logger.print(f'remote orig buffer {remote_orig_buffer} already exists! Exiting...')
            return
        else:
            collect_orig_latent_buffer(
                agent, orig_buffer_dir, Args.env_name, Adapt.latent_buffer_size,
                nstep=1, discount=Args.discount, frame_stack=Args.frame_stack,
                action_repeat=Args.action_repeat, seed=Args.seed, time_limit=Args.time_limit, Adapt=Adapt
            )

            logger.print(f'Compressing & Uploading the buffer: {remote_orig_buffer}')
            upload_and_cleanup(orig_buffer_dir, remote_orig_buffer, archive='tar')
            logger.print('Upload completed!')

    elif CollectData.buffer_type == 'target':
        buffer_prefix = get_buffer_prefix(
            Args.env_name, Args.action_repeat, Args.seed, eval_env=Args.eval_env_name,
            augment=Adapt.augment,
            distraction_types=Adapt.distraction_types,
            snapshot_prefix=Adapt.snapshot_prefix,
            distraction_intensity=Adapt.distraction_intensity
        )
        targ_buffer = Path(Args.tmp_dir) / 'data-collection' / buffer_prefix / 'workdir' / "targ_obs_buffer"
        logger.print('targ_buffer_dir', targ_buffer)
        remote_targ_buffer = pjoin(Args.checkpoint_root, buffer_prefix, 'targ_obs_buffer.tar')
        logger.print('remote targ_buffer', remote_targ_buffer)
        if mllogger.glob(remote_targ_buffer):
            logger.print(f'remote target buffer {remote_targ_buffer} already exists! Exiting...')
            return
        else:
            collect_targ_obs_buffer(
                targ_buffer, Args.eval_env_name, Adapt.latent_buffer_size, nstep=1,
                discount=Args.discount, frame_stack=Args.frame_stack, action_repeat=Args.action_repeat,
                seed=Args.seed, time_limit=Args.time_limit, Adapt=Adapt
            )

            logger.print(f'Compressing & Uploading the buffer: {remote_targ_buffer}')
            upload_and_cleanup(targ_buffer, remote_targ_buffer, archive='tar')
            logger.print('Upload completed!')

    # elif CollectData.buffer_type == 'obs_rails':
    #     from .adapt import load_adpt_agent
    #     from iti.invr_thru_inf.utils import get_bc_buffer_prefix
    #     assert Args.eval_env_name is None, "Set Args.eval_env_name to None as it is not used anywhere"

    #     # Decide (local) buffer_dir and (remote) target_path
    #     _, task, difficulty = parse_env_name(Args.env_name)
    #     buff_prefix = get_bc_buffer_prefix(
    #         task, Args.action_repeat, Args.seed,
    #         augment=False,
    #         difficulty=difficulty,
    #         distraction_types=Adapt.distraction_types,
    #         snapshot_prefix=Adapt.snapshot_prefix,
    #         distraction_intensity=Adapt.distraction_intensity
    #     )

    #     # Decide (local) buffer_dir and (remote) target_path
    #     local_buffer = Path(Args.tmp_dir) / 'data-collection' / buff_prefix / 'workdir' / "obs_rails_buffer"
    #     remote_buffer = pjoin(Args.checkpoint_root, buff_prefix, 'obs_rails_buffer.tar')
    #     logger.print('local_buffer', local_buffer)
    #     logger.print('remote_buffer', remote_buffer)

    #     # Download source buffer (with state info)
    #     src_buff_prefix = get_bc_buffer_prefix(
    #         task, Args.action_repeat, Args.seed,
    #         augment=False,
    #         difficulty=difficulty,
    #         distraction_types=None,
    #         snapshot_prefix=Adapt.snapshot_prefix,
    #         distraction_intensity=None
    #     )
    #     source_remote_buffer = pjoin(Args.checkpoint_root, src_buff_prefix, 'obs_state_buffer.tar')
    #     # source_buf_remote_path = source_buf_remote_path.replace('/bac/', '/none-intsty0.0000/')
    #     # source_buf_remote_path = source_buf_remote_path.replace('/col/', '/none-intsty0.0000/')
    #     # source_buf_remote_path = source_buf_remote_path.replace('/cam/', '/none-intsty0.0000/')
    #     # source_buf_remote_path = source_buf_remote_path.replace('/bac-cam-col/', '/none-intsty0.0000/')

    #     assert mllogger.glob(source_remote_buffer), f'remote source buffer {source_remote_buffer} not found'
    #     logger.print('source_remote_buffer', source_remote_buffer)
    #     source_local_buffer = Path(Args.tmp_dir) / 'data-collection' / src_buff_prefix / 'workdir' / 'obs_state_buffer'
    #     logger.print(f'Downloading buffer from remote: {source_remote_buffer}')
    #     mllogger.download_dir(source_remote_buffer, to=source_local_buffer, unpack='tar')
    #     logger.print('Downloading buffer from remote... done')

    #     from iti.invr_thru_inf.replay_buffer import Replay
    #     source_state_replay = Replay(
    #         source_local_buffer, buffer_size=Adapt.latent_buffer_size,
    #         batch_size=Adapt.batch_size, num_workers=1,  # NOTE: Batch size does not matter at all
    #         nstep=1, discount=Args.discount, store_states=False
    #     )


    #     if mllogger.glob(remote_buffer):
    #         logger.print(f'remote buffer {remote_buffer} already exists. Exiting...')
    #         return
    #     else:
    #         from iti.invr_thru_inf.replay_buffer import Replay
    #         from iti.invr_thru_inf.env_helpers import get_env
    #         from iti.invr_thru_inf.collect_offline_data import collect_trajectories
    #         agent = load_adpt_agent(Args, Adapt)

    #         # agent, store_states=True, random_policy=False
    #         replay = Replay(
    #             local_buffer, buffer_size=Adapt.latent_buffer_size,
    #             batch_size=Adapt.batch_size, num_workers=1,  # NOTE: Batch size does not matter at all
    #             nstep=1, discount=Args.discount, store_states=False
    #         )
    #         distr_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed,
    #                             distraction_config=Adapt.distraction_types, rails_wrapper=True,
    #                             intensity=Adapt.distraction_intensity)
    #         collect_trajectories(
    #             agent=None, env=distr_env, buffer_dir=local_buffer, num_steps=Adapt.latent_buffer_size // Args.action_repeat,
    #             pretrain_last_step=None, replay=replay, store_states=False, augment=None, action_repeat=Args.action_repeat,
    #             video_key='obs_rails_trajectory', encode_obs=False, random_policy=True,
    #             time_limit=Args.time_limit, src_state_replay=source_state_replay
    #         )

    #         logger.print(f'Compressing & Uploading the buffer: {remote_buffer}')
    #         upload_and_cleanup(local_buffer, remote_buffer, archive='tar')
    #         logger.print('Upload completed!')

    # elif CollectData.buffer_type == 'obs_state':
    #     from .adapt import load_adpt_agent
    #     from iti.invr_thru_inf.utils import get_bc_buffer_prefix
    #     assert Args.eval_env_name is None, "Set Args.eval_env_name to None as it is not used anywhere"

    #     # Decide (local) buffer_dir and (remote) target_path
    #     buff_prefix = get_bc_buffer_prefix(
    #         Args.env_name, Args.action_repeat, Args.seed,
    #         augment=Adapt.augment,
    #         distraction_types=Adapt.distraction_types,
    #         snapshot_prefix=Adapt.snapshot_prefix,
    #         distraction_intensity=Adapt.distraction_intensity
    #     )
    #     local_buffer = Path(Args.tmp_dir) / 'data-collection' / buff_prefix / 'workdir' / "obs_state_buffer"
    #     remote_buffer = pjoin(Args.checkpoint_root, buff_prefix, 'obs_state_buffer.tar')
    #     logger.print('local buffer', local_buffer)
    #     logger.print('remote buffer', remote_buffer)

    #     if mllogger.glob(remote_buffer):
    #         logger.print(f'remote buffer {remote_buffer} already exists. Exiting...')
    #         return
    #     else:
    #         from iti.invr_thru_inf.replay_buffer import Replay
    #         from iti.invr_thru_inf.env_helpers import get_env
    #         from iti.invr_thru_inf.collect_offline_data import collect_trajectories
    #         agent = load_adpt_agent(Args, Adapt)

    #         # TODO: Add flag for (1) sample action; (2) random action prob
    #         # TODO: Overwrite agent

    #         # agent, store_states=True, random_policy=False
    #         replay = Replay(
    #             local_buffer, buffer_size=Adapt.latent_buffer_size,
    #             batch_size=Adapt.batch_size, num_workers=1,  # NOTE: Batch size does not matter at all
    #             nstep=1, discount=Args.discount, store_states=False
    #         )
    #         distr_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed,
    #                             distraction_config=Adapt.distraction_types, rails_wrapper=False,
    #                             intensity=Adapt.distraction_intensity)
    #         collect_trajectories(
    #             agent=agent, env=distr_env, buffer_dir=local_buffer, num_steps=Adapt.latent_buffer_size // Args.action_repeat,
    #             pretrain_last_step=None, replay=replay , store_states=True, augment=None, action_repeat=Args.action_repeat,
    #             video_key='target_obs_trajectory', encode_obs=False, random_policy=False,
    #             time_limit=Args.time_limit
    #         )

    #         logger.print(f'Compressing & Uploading the buffer: {remote_buffer}')
    #         upload_and_cleanup(local_buffer, remote_buffer, archive='tar')
    #         logger.print('Upload completed!')
    else:
        raise ValueError('invalid buffer type', CollectData.buffer_type)

if __name__ == '__main__':
    # Parse arguments
    import argparse
    from ila.invr_thru_inf.config import Adapt, CollectData
    from .config import Args
    from ila.helpers.tticslurm import prepare_launch

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str,
                        help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    from params_proto.neo_hyper import Sweep
    sweep = Sweep(Args, Adapt, CollectData).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    # Set prefix
    job_name = kwargs['job_name']
    from ila import RUN
    RUN.prefix = f'{RUN.project}/{job_name}'

    prepare_launch(Args.job_name)

    # Obtain / generate wandb_runid
    params_path = pjoin(Args.checkpoint_root, RUN.prefix, 'parameters.pkl')
    try:
        wandb_runid = mllogger.read_params("wandb_runid", path=params_path)
    except Exception as e:
        print(e)
        wandb_runid = wandb.util.generate_id()
        mllogger.log_params(wandb_runid=wandb_runid, path=params_path)

    logger.info('params_path', params_path)
    logger.info('If wandb initialization fails with "conflict" error, deleting ^ forces to generate a new wandb id (but all intermediate progress will be lost.)')
    logger.print('wandb_runid:', wandb_runid)

    # Initialize wandb
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=f'{RUN.project}-datagen',
        # Track hyperparameters and run metadata
        config=kwargs,
        id=wandb_runid
    )

    wandb.run.summary['completed'] = wandb.run.summary.get('completed', False)

    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')

    # Controlled exit before slurm kills the process
    try:
        main(**kwargs)
        wandb.finish()
    except TimeoutError as e:
        wandb.mark_preempting()
        raise e
