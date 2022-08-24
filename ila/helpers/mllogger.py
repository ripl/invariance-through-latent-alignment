#!/usr/bin/env python3
"""Apply bunch of patches to ml-logger.
Mostly to get it work with URL prefixes like "file://" or "gs://"
"""

from tempfile import NamedTemporaryFile


def patch_glob(logger):
    """Support file://, gs://, and s3:// prefix"""
    # NOTE:This is necessary, if you return logger.glob(...) in __glob,
    # that'll cause infinite recursion, as logger.glob points to the overwritten method
    original_glob = logger.glob
    def __glob(query, wd=None, recursive=True, start=None, stop=None):
        if query.startswith('file://'):
            from glob import glob
            return glob(query[7:])
        elif query.startswith('s3://'):
            return logger.glob_s3(query[5:], wd=wd)
        elif query.startswith('gs://'):
            return logger.glob_gs(query[5:], wd=wd)
        # elif not wd and query.startswith('/'):
        #     return ['/' + p for p in
        #             logger.client.glob(query, wd="/", recursive=recursive, start=start, stop=stop)]

        # wd = pJoin(__prefix__, wd or "")
        # return logger.client.glob(query, wd=wd, recursive=recursive, start=start, stop=stop)
        return original_glob(query, wd, recursive, start, stop)
    return __glob

def patch_upload_file(logger):
    """Support file://, gs://, and s3:// prefix"""
    import cloudpickle
    from tempfile import NamedTemporaryFile
    original_upload_file = logger.upload_file
    def __upload_file(file_path: str = None, target_path: str = "files/", once=True):
        if target_path.startswith('s3://'):
            logger.upload_s3(str(file_path), path=target_path[5:])
        elif target_path.startswith('gs://'):
            logger.upload_gs(str(file_path), path=target_path[5:])
        elif target_path.startswith('file://'):
            import shutil
            shutil.copyfile(file_path, target_path[7:])
        else:
            original_upload_file(file_path, target_path, once=once)
    return __upload_file


def patch_save_pkl(logger):
    """Support file://, gs://, and s3:// prefix"""
    import os
    # NOTE: logger.log_params calls logger.save_pkl this internally.
    original_save_pkl = logger.save_pkl
    def __save_pkl(data, path=None, append=False, use_dill=False):
        from tempfile import NamedTemporaryFile
        import pickle
        if path.startswith('file://'):
            os.makedirs(os.path.dirname(path[7:]), exist_ok=True)
            with open(path[7:], 'wb') as f:
                pickle.dump(data, f)
        elif path.startswith('s3://') or path.startswith('gs://'):
            # NOTE: TempFile cannot work with a large file (20GB+)
            with NamedTemporaryFile(suffix=f'.pkl') as ntp:
                pickle.dump(data, ntp)
                logger.upload_file(ntp.name, path)
        else:
            original_save_pkl

    return __save_pkl


def patch_download_file(logger):
    """Support file://, gs://, and s3:// prefix"""
    original_download_file = logger.download_file
    def __download_file(*keys, path=None, to, relative=False):
        if path.startswith('file://'):
            import shutil
            shutil.copy(path[7:], to)
        elif path.startswith('gs://'):
            logger.download_gs(path[5:], to=to)
        elif path.startswith('s3://'):
            logger.download_s3(path[5:], to=to)
        else:
            original_download_file(*keys, path=path, to=to, relative=relative)

    return __download_file


def patch_load_pkl(logger):
    """Support file://, gs://, and s3:// prefix"""
    # NOTE: logger.read_params uses logger.load_pkl.
    original_load_pkl = logger.load_pkl
    def __load_pkl(*keys, start=None, stop=None, tries=1, delay=1):
        import pickle
        # NOTE: this takes an additional kwarg: path
        # NOTE: This is weird, but the original load_pkl doesn't take path argument.
        # This is because logger.read_params('hoge', path='/path/to/params.pkl') calls
        # logger.load_pkl(logger.glob(path)[0] if "*" in path else path) internally.
        # Thus we'll overwrite behavior only when len(keys) == 1 and keys[0].startswith('...')
        if len(keys) == 1 and (keys[0].startswith('file://') or keys[0].startswith('s3://') or keys[0].startswith('gs://')):
            assert start is None and stop is None and tries==1 and delay==1
            path = keys[0]
            if not logger.glob(path):
                return
            with NamedTemporaryFile(suffix='.pkl') as ntp:
                logger.download_file(path=path, to=ntp.name)
                return [pickle.load(ntp)]
        else:
            return original_load_pkl(*keys, start=start, stop=stop, tries=tries, delay=delay)

    return __load_pkl

def patch_get_parameters(logger):
    """Return None rather than FileNotFoundError if the corresponding file is not found."""
    original_get_parameters = logger.get_parameters
    def __get_parameters(*keys, path="parameters.pkl", silent=False, **kwargs):
        try:
            return original_get_parameters(*keys, path=path, silent=silent, **kwargs)
        except FileNotFoundError as e:
            print(f'(logger.read_params) Captured:{e}\n'
                  f'(logger.read_params) key:{keys}')
            return
    return __get_parameters


def patch_save_torch(logger):
    original_save_torch = logger.save_torch
    def __save_torch(obj, *keys, path=None, tries=3, backup=3.0):
        if path.lower().startswith('s3://') or path.lower().startswith('gs://') or path.lower().startswith('file://'):
            from os.path import join as pjoin
            path = pjoin(*keys, path)

            from pathlib import Path
            import shutil
            import torch
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(delete=True) as tfile:
                torch.save(obj, tfile)

                if path.lower().startswith('s3://'):
                    tfile.seek(0)
                    return logger.upload_s3(source_path=tfile.name, path=path[5:])
                elif path.lower().startswith('gs://'):
                    tfile.seek(0)
                    return logger.upload_gs(source_path=tfile.name, path=path[5:])
                elif path.lower().startswith('file://'):
                    tfile.seek(0)
                    path = path[7:]
                    parent_dir = Path(path).resolve().parents[0]
                    if not parent_dir.is_dir():
                        parent_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(tfile.name, path)
                    return
        else:
            return original_save_torch(obj, *keys, path=path, tries=tries, backup=backup)
    return __save_torch

def patch_load_torch(logger):
    original_load_torch = logger.load_torch
    def __load_torch(*keys, path=None, map_location=None, **kwargs):
        import os
        from os.path import join as pjoin
        import torch, tempfile
        if path.lower().startswith('s3://') or path.lower().startswith('gs://') or path.lower().startswith('file://'):
            path = pjoin(*keys, path)
            if path.lower().startswith('s3://'):
                postfix = os.path.basename(path)
                with tempfile.NamedTemporaryFile(suffix=f'.{postfix}') as ntp:
                    logger.download_s3(path[5:], to=ntp.name)
                    return torch.load(ntp.name, map_location=map_location, **kwargs)
            elif path.lower().startswith('gs://'):
                postfix = os.path.basename(path)
                with tempfile.NamedTemporaryFile(suffix=f'.{postfix}') as ntp:
                    logger.download_gs(path[5:], to=ntp.name)
                    return torch.load(ntp.name, map_location=map_location, **kwargs)
            elif path.lower().startswith('file://'):
                return torch.load(path[7:], map_location=map_location, **kwargs)
        else:
            original_load_torch(*keys, path=path, map_location=map_location, **kwargs)

    return __load_torch


def patch_logger(logger):
    logger.glob = patch_glob(logger)
    logger.upload_file = patch_upload_file(logger)
    logger.download_file = patch_download_file(logger)
    # logger.save_pkl = patch_save_pkl(logger)
    # logger.load_pkl = patch_load_pkl(logger)
    logger.get_parameters = patch_get_parameters(logger)
    logger.read_params = logger.get_parameters
    logger.save_torch = patch_save_torch(logger)
    logger.torch_save = logger.save_torch
    logger.load_torch = patch_load_torch(logger)
    logger.torch_load = logger.load_torch
    return logger

def test_patch_logger():
    from ml_logger import logger
    from learning_with_library.dmc_gen.config import Args
    logger = patch_logger(logger)

    logger.log_params(Args=vars(Args), path='file://parameters.pkl')
    loaded_args = logger.read_params("Args", path='file://parameters.pkl')
    print(loaded_args)

def get_ml_logger(prefix):
    from ml_logger import ML_Logger
    mllogger = ML_Logger(prefix=prefix)
    return patch_logger(mllogger)

if __name__ == '__main__':
    test_patch_logger()
