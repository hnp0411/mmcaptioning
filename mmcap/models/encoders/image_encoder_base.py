"""
Luke
"""
from abc import abstractmethod
import torch
import torch.nn as nn

from mmcv.utils import print_log

from mmcap.utils import get_root_logger
from ..builder import build_backbone
from ..utils import PositionEmbeddingSine


class ImageEncoderBase(nn.Module):
    """Base Image Encoder for Image Captioning

    Image Backbone + Mask Reductor + 2D Position Embedding
    """

    def __init__(self,
                 backbone,
                 backbone_feats=2048,
                 pos_feats=128,
                 pos_temperature=10000,
                 pos_norm=True):

        super(ImageEncoderBase, self).__init__()

        self.backbone = build_backbone(backbone)
        self.backbone_feats = backbone_feats
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=pos_feats,
                                                        temperature=pos_temperature,
                                                        normalize=pos_norm)
        self.backbone_input_projection(backbone_feats,
                                       pos_feats * 2,
                                       kernel_size=1)
        
    def backbone_input_projection(self, 
                                  src_feats, 
                                  dst_feats,
                                  kernel_size=1):
        """Input Projection to Decoder

        """
        self.input_proj = nn.Conv2d(src_feats,
                                    dst_feats,
                                    kernel_size)

    def init_weights(self, pretrained=None):
        """Initialize the weights in Encoder

        """
        if pretrained is not None:
            assert isinstance(pretrained, dict)
            logger = get_root_logger()
            print_log(f'load encoder weight from: {pretrained}',
                      logger=logger)
            self.backbone.init_weights(pretrained=pretrained['backbone_pretrained'])

    @abstractmethod
    def forward(self, img, img_mask):
        """Forward
        
        Args:
            img
            img_mask

        Outputs:
            img_feature
            reduced_img_mask
            img_positional_embedding
        """
        pass
