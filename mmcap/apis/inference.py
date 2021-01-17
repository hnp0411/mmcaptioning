"""
Luke
"""
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmcap.datasets.pipelines import Compose
from mmcap.models import build_caption


def init_caption(config, device='cuda:1', checkpoint=None):
    """Initialize a caption model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint: checkpoint path

    Returns:
        nn.Module: The constructed caption model.
    """
    config = mmcv.Config.fromfile(config)

    # Includes init weights
    model = build_caption(config.model)

    model.cfg = config  # save the config in the model for convenience

    # load_checkpoint if checkpoint is not None
    if checkpoint is not None:
        print('\nLoad Checkpoint From ...\n    {}\n'.format(checkpoint))
        load_checkpoint(model, checkpoint, strict=False, logger=None,
                        map_location=lambda storage, loc: storage)

    model.to(device)
    model.eval()
    return model


def generate_caption(model, 
                     tokenizer, 
                     tokenizer_cfg,
                     img, 
                     generate_method='greedy'):
    """Generate Captions of given img path.

    Args:
        model (nn.Module): Image Captioning Model
        tokenizer: For preprocess pipeline, generate text
        img (str): img file path

    Returns:
        Generated text result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare data
    # Add dummy empty caption
    cap_info = dict(caption='',
                    tokenizer=tokenizer,
                    tokenizer_cfg=tokenizer_cfg)
    data = dict(img_info=dict(filename=img),
                img_prefix=None,
                cap_info=cap_info)

    # build the data pipeline
    tokenizer_type = tokenizer_cfg['type']
    test_pipeline = Compose(build_caption_test_pipeline(tokenizer_type))

    # preprocess data
    data = test_pipeline(data)

    for key in data:
        data[key] = data[key]._data
        if key in ['img', 'img_mask']:
            data[key] = data[key][None]
        if key != 'img_metas':
            data[key] = data[key].to(device='cuda:1')

#  TODO : support distributed inference for multiple images
#    data = collate([data], samples_per_gpu=1)
#
#    if next(model.parameters()).is_cuda:
#        # scatter to specified GPU
#        data = scatter(data, [device])[0]
    
    # set decode config
    data['decoding_cfg'] = cfg.test_cfg.decoding_cfg
    # forward the model
    result = model.generate_caption(**data)
    print('\nGenerated Vector : {}'.format(result))

    # postprocessing
    result = result[0].tolist()
    for ind, token in enumerate(result):
        if token == 3:
            result = result[:ind+1]
            break

    # get result
    caption = tokenizer.decode(result, skip_special_tokens=True)

    return caption


def build_caption_test_pipeline(tokenizer_type:str):
    """Build captioning pipeline for captioning by tokenizer type.

    """
    if 'Wrapper' in tokenizer_type: # huggingface tokenizer
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        max_dim = 299
        
        test_pipeline=[
            # preprocess img
            dict(type='LoadPILImage'),
            dict(type='UnderMax', max_dim=max_dim),
            dict(type='ToTensorCaptioning', keys = ['img']),
            dict(type='NormalizeCaptioning'),
            # create img_mask
            dict(type='CreateImgMask', max_dim=max_dim),
            # preprocess caption
            dict(type='EncodeEmptyCaption',
                 caption_max_length=128,
                 padding='max_length',
                 truncation=True),
            dict(type='DefaultFormatBundleCaptioning'),
            dict(type='Collect',
                 keys=['img', 'img_mask', 'cap', 'cap_mask'],
                 meta_keys=['filename'])
        ]
    
    return test_pipeline
