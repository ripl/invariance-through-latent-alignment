#!/usr/bin/env python3

def main(**kwargs):
    import os
    import torch
    from os.path import join as pJoin
    from pathlib import Path
    from .config import Args, Adapt
    from ml_logger import ML_Logger

    Args._update(kwargs)
    Adapt._update(kwargs)

    # TODO Check if the training is completed
    logger = ML_Logger(prefix=Adapt.snapshot_prefix)

    try:
        completion_time = mllogger.read_params('job.completionTime')
        logger.print(f'job.completionTime is set:{completion_time}\n', __prefix__)
        logger.print('removing completion entry...')
        params = logger.load_pkl('parameters.pkl')

    except (KeyError, TypeError) as e:
        # Load from local file
        raise RuntimeError('completion time is not found. Training is not done yet.', __prefix__)

    new_params = []
    for param in params:
        try:
            if param['job']['status'] == 'completed':
                continue
        except KeyError:
            pass
        new_params.append(param)

    for i, new_param in enumerate(new_params):
        logger.save_pkl(new_param, 'parameters.pkl', append=(i != 0))

