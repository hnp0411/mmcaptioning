"""
Luke
"""
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32

from mmcap.utils import get_root_logger
from . import CaptionBase
from ..builder import CAPTIONS, build_decoder
from ..utils import topktopp


@CAPTIONS.register_module()
class ResNetTransformer(CaptionBase):
    """Image Captioning Model.

    Use ResNet (Imagenet pretrained)
    + Transformer
    """
    def __init__(self,
                 encoder,
                 decoder,
                 pretrained=None,
                 ffnn_hidden_dims=[512],
                 ffnn_num_layers=3):
        super(ResNetTransformer, self).__init__(encoder, 
                                                decoder, 
                                                pretrained, 
                                                ffnn_hidden_dims,
                                                ffnn_num_layers)

    def extract_feat(self):
        pass

    @force_fp32(apply_to=('pred', ))
    def build_loss(self, pred, gt):
        return nn.CrossEntropyLoss()(pred, gt)

    def forward_train(self,
                      img_metas,
                      img,
                      img_mask,
                      cap,
                      cap_mask,
                      **kwargs):

        img, img_mask, pos_embed = self.encoder(img, img_mask)
        hs = self.decoder(img, img_mask, pos_embed, cap[:,:-1], cap_mask[:,:-1])
        out = self.ffnn(hs.permute(1, 0, 2))

        losses = dict()
        losses['loss'] = self.loss(out.permute(0, 2, 1), cap[:, 1:])
        out = self.to_fp32(out)

        return losses, out

    def simple_test(self,
                    img_metas,
                    img,
                    img_mask,
                    cap,
                    cap_mask,
                    **kwargs):

        img, img_mask, pos_embed = self.encoder(img, img_mask)
        hs = self.decoder(img, img_mask, pos_embed, cap[:,:-1], cap_mask[:,:-1])
        out = self.ffnn(hs.permute(1, 0, 2))

        losses = dict()
        losses['loss'] = self.loss(out.permute(0, 2, 1), cap[:,1:])
        out = self.to_fp32(out)

        return losses, out

    def generate_greedy_caption(self,
                                img_metas,
                                img,
                                img_mask,
                                cap,
                                cap_mask,
                                **kwargs):
        """Generate caption result by greedy search

        """
        cap, cap_mask = cap[:, :-1], cap_mask[:, :-1]

        with torch.no_grad():
            self.eval()
            img, img_mask, pos = self.encoder(img, img_mask)
            #for ind in range(self.cfg.model.decoder.max_position_embeddings-1):
            for ind in range(127):
                hs = self.decoder(img,
                                  img_mask,
                                  pos,
                                  cap,
                                  cap_mask)
                pred = self.ffnn(hs.permute(1, 0, 2))
                pred = self.to_fp32(pred)
                pred = pred[:, ind, :]
                pred_id = torch.argmax(pred, axis=-1)

                cap[:, ind+1] = pred_id[0]
                cap_mask[:, ind+1] = False

        # TODO : refactoring
                if pred_id[0] == 3:
                    break

        return cap[:,:ind+2]

    @auto_fp16(apply_to=('img', ))
    def generate_topktopp_caption(self,
                                  img_metas,
                                  img,
                                  img_mask,
                                  cap,
                                  cap_mask,
                                  topk=5,
                                  topp=0.8,
                                  **kwargs):
        """Generate caption result by topk-topp decoding

        """
        cap, cap_mask = cap[:, :-1], cap_mask[:, :-1]

        with torch.no_grad():
            self.eval()
            img, img_mask, pos = self.encoder(img, img_mask)
            #for ind in range(self.cfg.model.decoder.max_position_embeddings-1):
            for ind in range(127):
                hs = self.decoder(img,
                                  img_mask,
                                  pos,
                                  cap,
                                  cap_mask)
                pred = self.ffnn(hs.permute(1, 0, 2))
                pred = self.to_fp32(pred)
                pred = pred[:, ind, :]

                # top K - top P results
                pred_id = topktopp(pred, topk, topp)

                cap[:, ind+1] = pred_id[0]
                cap_mask[:, ind+1] = False

        # TODO : refactoring
                if pred_id[0] == 3:
                    break

        return cap[:,:ind+2]
