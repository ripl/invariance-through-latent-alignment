#!/usr/bin/env python3
from warnings import simplefilter  # noqa
simplefilter(action='ignore', category=DeprecationWarning)

import torch


def main(**kwargs):
    from warnings import simplefilter  # noqa

    from ml_logger import RUN, ML_Logger, logger

    from ila.invr_thru_inf.adapt import adapt_offline, prepare_buffers
    from ila.invr_thru_inf.config import Adapt
    from ila.invr_thru_inf.env_helpers import get_env
    from ila.invr_thru_inf.replay_buffer import Replay
    from ila.invr_thru_inf.utils import (get_distr_string,
                                         set_seed_everywhere, update_args, record_video)
    from ila.invr_thru_inf.adapt import load_snapshot, Progress
    from ila.helpers.tticslurm import set_egl_id
    from .adapt import load_adpt_agent

    from .config import Args, Agent
    simplefilter(action='ignore', category=DeprecationWarning)

    if not torch.cuda.is_available():
        raise RuntimeError('torch.cuda.is_available() is False!!')

    logger.start('run', 'start')

    # Update config parameters based on kwargs
    set_egl_id()
    set_seed_everywhere(kwargs['seed'])

    Args._update(kwargs)
    Agent._update(kwargs)
    Adapt._update(kwargs)


    # ===== make or load adaptation-agent =====
    try:
        custom_prefix = __prefix__.replace('run_record_video/videos-cherrypick', 'run/noaug-all')
        src_logger = ML_Logger(prefix=custom_prefix)
        logger.print(f'prefix {src___prefix__}')

        # Adaptation has been already completed
        completion_time = src_mllogger.read_params('job.completionTime')
        logger.print(f'job.completionTime is set:{completion_time}\n',
                     'This job seems to have been completed already.')

        pret_agent = load_adpt_agent(Args, Agent, Adapt)
        adapted_agent = load_snapshot(Args.checkpoint_root, custom_prefix=custom_prefix)

        # Load Progress from checkpoint.pkl
        assert src_mllogger.glob('checkpoint.pkl')
        Progress._update(src_mllogger.read_params(path="checkpoint.pkl"))

    except Exception as e:
        logger.print('This adaptation has not been finished yet!')
        logger.print(__prefix__)
        raise e


    source_env = get_env(Args.train_env, Args.frame_stack, Args.action_repeat, Args.seed + 1000)

    target_env = get_env(Args.eval_env, Args.frame_stack, Args.action_repeat, Args.seed,
                         distraction_config=Adapt.distraction_types,
                         intensity=Adapt.distraction_intensity)

    record_video(Adapt, pret_agent, source_env,
                 progress=Progress, video_prefix='videos/source')
    record_video(Adapt, pret_agent, target_env,
                 progress=Progress, video_prefix='videos/zeroshot',
                 cherry_pick='min')
    record_video(Adapt, adapted_agent, target_env,
                 progress=Progress, video_prefix='videos/target',
                 cherry_pick='max')
