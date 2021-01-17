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
from . import init_caption


def extract_encoder_feat(model, tokenizer, img):
    """Extract encoder features of caption model.

    Args:
        model (nn.Module): Image Captioning Model
        tokenizer: For preprocess pipeline
        img (str): img file path

    Returns:
        Extracted feature result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    # Add dummy caption
    cap_info = dict(caption='',
                    tokenizer=tokenizer)
    data = dict(img_info=dict(filename=img), 
                img_prefix=None,
                cap_info=cap_info)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        # img, img_mask, pos
        result = model.extract_feat(data['img'][0], data['img_mask'][0])
        
        #result = model(return_loss=False, rescale=True, **data)[0]
    return result
