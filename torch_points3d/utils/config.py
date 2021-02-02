import numpy as np
from typing import List
import shutil
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
import logging
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from .enums import ConvolutionFormat
from torch_points3d.utils.debugging_vars import DEBUGGING_VARS
from torch_points3d.utils.colors import COLORS, colored_print
import subprocess

log = logging.getLogger(__name__)


class ConvolutionFormatFactory:
    @staticmethod
    def check_is_dense_format(conv_type):
        if (
            conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower()
            or conv_type.lower() == ConvolutionFormat.MESSAGE_PASSING.value.lower()
            or conv_type.lower() == ConvolutionFormat.SPARSE.value.lower()
        ):
            return False
        elif conv_type.lower() == ConvolutionFormat.DENSE.value.lower():
            return True
        else:
            raise NotImplementedError("Conv type {} not supported".format(conv_type))


class Option:
    """This class is used to enable accessing arguments as attributes without having OmaConf.
       It is used along convert_to_base_obj function
    """

    def __init__(self, opt):
        for key, value in opt.items():
            setattr(self, key, value)


def convert_to_base_obj(opt):
    return Option(OmegaConf.to_container(opt))


def set_debugging_vars_to_global(cfg):
    for key in cfg.keys():
        key_upper = key.upper()
        if key_upper in DEBUGGING_VARS.keys():
            DEBUGGING_VARS[key_upper] = cfg[key]
    log.info(DEBUGGING_VARS)


def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)


def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)


def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)


def create_symlink_from_eval_to_train(eval_checkpoint_dir):
    root = os.path.join(os.getcwd(), "evals")
    if not os.path.exists(root):
        os.makedirs(root)
    num_files = len(os.listdir(root)) + 1
    os.symlink(eval_checkpoint_dir, os.path.join(root, "eval_{}".format(num_files)))


def get_from_kwargs(kwargs, name):
    module = kwargs[name]
    kwargs.pop(name)
    return module


def fetch_arguments_from_list(opt, index, special_names):
    """Fetch the arguments for a single convolution from multiple lists
    of arguments - for models specified in the compact format.
    """
    args = {}
    for o, v in opt.items():
        name = str(o)
        if is_list(v) and len(getattr(opt, o)) > 0:
            if name[-1] == "s" and name not in special_names:
                name = name[:-1]
            v_index = v[index]
            if is_list(v_index):
                v_index = list(v_index)
            args[name] = v_index
        else:
            if is_list(v):
                v = list(v)
            args[name] = v
    return args


def flatten_compact_options(opt):
    """Converts from a dict of lists, to a list of dicts"""
    flattenedOpts = []
    for index in range(int(1e6)):
        try:
            flattenedOpts.append(
                DictConfig(fetch_arguments_from_list(opt, index)))
        except IndexError:
            break
    return flattenedOpts


def fetch_modalities(opt, modality_names):
    """Search for supported modalities in the compact format config."""
    modalities = []
    for o, v in opt.items():
        name = str(o).lower()
        if name not in modality_names:
            continue
        assert hasattr(v, 'down_conv') \
               and hasattr(v, 'atomic_pooling') \
               and hasattr(v, 'view_pooling') \
               and hasattr(v, 'fusion') \
               and hasattr(v, 'branching_index'), \
            f"Found '{name}' modality in the config but could not " \
            f"recover all required attributes: ['down_conv' " \
            f"'atomic_pooling', 'view_pooling', 'fusion', 'branching_index]"
        modalities.append(name)
    return modalities
