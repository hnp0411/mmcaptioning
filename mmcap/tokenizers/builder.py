"""
Luke
"""
from mmcv.utils import Registry, build_from_cfg
from torch import nn

TOKENIZERS = Registry('tokenizer')

def build_tokenizer(cfg, default_args=None):

    if 'Wrapper' in cfg['type']:
        wrapper = build_from_cfg(cfg, TOKENIZERS, default_args)
        return wrapper.tokenizer

    else:
        return build_from_cfg(cfg, TOKENIZERS, default_args)

