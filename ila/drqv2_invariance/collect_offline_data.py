#!/usr/bin/env python3
from os.path import join as pJoin
from pathlib import Path

from ila.invr_thru_inf.collect_offline_data import (
    upload_and_cleanup, collect_orig_latent_buffer, collect_targ_obs_buffer
)


def main(**kwargs):
    from ml_logger import logger, RUN, ML_Logger
    from .config import Args, Agent
    from ila.invr_thru_inf.config import Adapt, CollectData
    from ila.invr_thru_inf.utils import set_seed_everywhere, get_buffer_prefix
    from ila.helpers.tticslurm import set_egl_id

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    logger.start('run')
    set_egl_id()
    set_seed_everywhere(kwargs['seed'])

    # Update config parameters based on kwargs
    Args._update(kwargs)
    Agent._update(kwargs)
    Adapt._update(kwargs)

    if Adapt.snapshot_prefix:
        src_logger = ML_Logger(
            prefix=Adapt.snapshot_prefix
        )

        assert src___prefix__ != __prefix__
        logger.print('src___prefix__', src___prefix__)
        logger.print('__prefix__', __prefix__)

        # Check if the job is completed
        try:
            completion_time = src_mllogger.read_params('job.completionTime')
            logger.print(f'Pre-training was completed at {completion_time}.')
        except KeyError:
            logger.print(f'training for {__prefix__} has not been finished yet.')
            logger.print('job.completionTime was not found.')
            if not RUN.debug:
                raise RuntimeError

        # Load from the checkpoint
        assert RUN.debug or src_mllogger.glob('checkpoint.pkl')

        # Update parameters
        keep_args = ['eval_env', 'seed', 'tmp_dir', 'checkpoint_root', 'time_limit', 'device']
        Args._update({key: val for key, val in src_mllogger.read_params("Args").items() if key not in keep_args})
        Agent._update(**src_mllogger.read_params('Agent'))

    mllogger.log_params(Args=vars(Args), Agent=vars(Agent), Adapt=vars(Adapt))

    # Update config parameters based on kwargs
    CollectData._update(kwargs)


    # ===== Prepare the buffers =====

    # Currently only this configuration is suppoorted.
    assert Adapt.policy_on_clean_buffer == "random" and Adapt.policy_on_distr_buffer == "random"

    if CollectData.buffer_type == 'original':
        from .adapt import load_adpt_agent
        buffer_dir_prefix = Path(Args.tmp_dir) / 'data-collection' / get_buffer_prefix(
            Args.train_env, Args.action_repeat, Args.seed, eval_env=Args.eval_env,
            latent_buffer=True, **Adapt
        )

        orig_buffer_dir = buffer_dir_prefix / 'workdir' / "orig_latent_buffer"
        logger.print('orig_buffer_dir', orig_buffer_dir)
        agent = load_adpt_agent(Args, Agent, Adapt)
        target_path = pJoin(Args.checkpoint_root, get_buffer_prefix(
            Args.train_env, Args.action_repeat, Args.seed, eval_env=Args.eval_env,
            latent_buffer=True, **Adapt), 'orig_latent_buffer.tar')
        logger.print(target_path)

        if not mllogger.glob(target_path):
            collect_orig_latent_buffer(
                agent, orig_buffer_dir, Args.eval_env, Adapt.latent_buffer_size,
                nstep=1, discount=Args.discount, frame_stack=Args.frame_stack,
                action_repeat=Args.action_repeat, seed=Args.seed, time_limit=Args.time_limit, Adapt=Adapt
            )

            logger.print(f'Compressing & Uploading the buffer: {target_path}')
            upload_and_cleanup(orig_buffer_dir, target_path, archive='tar')
            logger.print('Upload completed!')

    elif CollectData.buffer_type == 'target':
        buffer_dir_prefix = Path(Args.tmp_dir) / 'data-collection' / get_buffer_prefix(
            Args.train_env, Args.action_repeat, Args.seed, eval_env=Args.eval_env, **Adapt
        )
        targ_buffer_dir = buffer_dir_prefix / 'workdir' / "targ_obs_buffer"
        logger.print('targ_buffer_dir', targ_buffer_dir)
        target_path = pJoin(Args.checkpoint_root, get_buffer_prefix(
            Args.train_env, Args.action_repeat, Args.seed, eval_env=Args.eval_env, **Adapt
        ), 'targ_obs_buffer.tar')
        if not mllogger.glob(target_path):
            collect_targ_obs_buffer(
                targ_buffer_dir, Args.eval_env, Adapt.latent_buffer_size, nstep=1,
                discount=Args.discount, frame_stack=Args.frame_stack, action_repeat=Args.action_repeat,
                seed=Args.seed, time_limit=Args.time_limit, Adapt=Adapt
            )

            logger.print(f'Compressing & Uploading the buffer: {target_path}')
            upload_and_cleanup(targ_buffer_dir, target_path, archive='tar')
            logger.print('Upload completed!')

    else:
        raise ValueError('invalid buffer type', CollectData.buffer_type)

    logger.print('=========== Completed!!! ==============')


if __name__ == '__main__':
    main()
