import os
from os import path as osp
from .logger import AvgTimer, MessageLogger, get_root_logger, init_tb_logger, init_wandb_logger
from .dist_util import master_only, get_dist_info, init_dist
from .misc import check_resume, get_time_str, make_exp_dirs, mkdir_and_rename, scandir, set_random_seed, sizeof_fmt
from .options import ordered_yaml, yaml_load, dict2str, parse_options, copy_opt_file
from .file_client import FileClient
from .img_util import imfrombytes, img2tensor
from .color_util import rgb2ycbcr_pt

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)