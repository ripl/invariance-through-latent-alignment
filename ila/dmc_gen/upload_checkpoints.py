#!/usr/bin/env python3

def glob_s3(path):
    from ml_logger import logger
    assert path.startswith('s3://')
    try:
        flist = mllogger.glob_s3(path[5:])
    except KeyError:
        flist = []
    return flist


def main(**kwargs):
    import torch
    from os.path import join as pJoin
    from pathlib import Path
    from .config import Args
    from ml_logger import logger, ML_Logger

    Args._update(kwargs)

    # Load from local file
    # NOTE: Args.snapshot_dir did point to "/share/data/ripl/takuma/snapshots"
    # snapshot = pJoin(Args.snapshot_dir, Adapt.snapshot_prefix, 'snapshot.pt')
    # snapshot_src = pJoin(Args.tmp_dir, Adapt.snapshot_prefix, 'snapshot.pt')
    prefix = 's3://ge-data-improbable/drqv2_invariance'
    snapshot_target = pJoin(prefix, Args.snapshot_prefix, 'actor_state.pt')

    # TODO: Check if the file is already uploaded
    if glob_s3(snapshot_target):
        logger.print('snapshot is already available!', snapshot_target)
        return

    # logger.print('Loading from', snapshot_src)
    # agent = mllogger.load_torch(path=snapshot_src)
    # logger.print('Loading is done.')
    
    # NOTE: (dmc_gen) This assumes the snapshot is available at ML-dash server
    src_logger = ML_Logger(prefix=Args.snapshot_prefix)
    agent = src_mllogger.load_torch('snapshot.pt')

    mllogger.save_torch(agent.actor.state_dict(), path=snapshot_target)

    logger.print('Uploaded to:', snapshot_target)

    # from tempfile import NamedTemporaryFile
    # with NamedTemporaryFile(delete=True) as tfile:
    #     logger.print('Downloading snapshot...')
    #     src_logger.download_file('snapshot.pt', to=tfile.name)

    #     logger.print('Uploading model to', snapshot_target)
    #     assert snapshot_target.startswith('s3://')
    #     logger.upload_s3(tfile.name, path=snapshot_target[5:])

    # mllogger.save_torch(agent, path=snapshot_target)
    logger.print('================= completed !!! ===================')
