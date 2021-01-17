"""
Luke
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16

from . import ResNetEncoder
from ..builder import ENCODERS


@ENCODERS.register_module()
class Res2NetEncoder(ResNetEncoder):
    """Image Encoder for image captioning.
     
    """
    
    def __init__(self,
                 backbone,
                 backbone_feats=2048,
                 pos_feats=128,
                 pos_temperature=10000,
                 pos_norm=True):

        super(Res2NetEncoder, self).__init__(backbone,
                                            backbone_feats,
                                            pos_feats,
                                            pos_temperature,
                                            pos_norm)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_mask):
        img = self.backbone(img)[-1]
        # cbnet backbone output
        # [batch_size, 256, 200, 336]
        # [batch_size, 512, 100, 168]
        # [batch_size, 1024, 50, 84]
        # [batch_size, 2048, 25, 42]
        img_mask = F.interpolate(img_mask.float(),
                                 size=img.shape[-2:]).to(torch.bool)
        pos_embed = self.position_embedding(img, img_mask)
        img = self.input_proj(img)
        return img, img_mask, pos_embed
