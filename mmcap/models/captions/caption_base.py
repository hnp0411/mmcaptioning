"""
Luke
"""
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import mmcv
from mmcv.utils import print_log
from mmcv.runner import auto_fp16, force_fp32

from mmcap.utils import get_root_logger

from ..builder import CAPTIONS, build_encoder, build_decoder


class CaptionBase(nn.Module):
    """Base Caption model for captions.

    Image Encoder + Text Decoder
    """

    def __init__(self,
                 encoder,
                 decoder,
                 pretrained=None,
                 ffnn_hidden_dims=[512],
                 ffnn_num_layers=3):

        super(CaptionBase, self).__init__()

        assert(len(ffnn_hidden_dims)==ffnn_num_layers-2)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.init_weights(pretrained)
        self.hidden_dim = decoder['hidden_dim']
        self.vocab_size = decoder['vocab_size']
        self.ffnn = FFNN(self.hidden_dim, 
                         ffnn_hidden_dims, 
                         self.vocab_size,
                         3)
        self.loss = self.build_loss
        self.fp16_enabled = False

    @abstractmethod
    @auto_fp16(apply_to=('img', )) # TODO : 정리
    def extract_feat(self, img, mask):
        """Extract CNN features from image.

        """
        pass

    @abstractmethod
    def build_loss(self, pred, gt):
        """Build Loss Function.

        """
        pass

    @abstractmethod
    def forward_train(self, 
                      img_metas, 
                      img,
                      img_mask,
                      cap,
                      cap_mask, 
                      **kwargs):
        """Forward Train
        Args:
            img_metas (list[dict]): List of image info dict.
                Currently it contains filename, raw caption information.
            img (Tensor): of shape (N, C, H, W) encoding input images.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    async def async_simple_test(self, 
                                img_metas,
                                img, 
                                img_mask, 
                                cap,
                                cap_mask,
                                **kwargs): 
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, 
                    img_metas, 
                    img, 
                    img_mask,
                    cap,
                    cap_mask,
                    **kwargs):
        """Test without augmentation.
        
        """
        pass

    def aug_test(self, 
                 imgs_metas, 
                 img, 
                 img_mask,
                 cap,
                 cap_mask,
                 **kwargs): 
        """Test with augmentations.

        """
        pass

    @force_fp32(apply_to=('out', ))
    def to_fp32(self, out):
        """Convert output to fp32 when mixed precision training.

        """
        return out

    def init_weights(self, pretrained:dict): 
        """Initialize the weights in Captioning Model.

        Args:
            pretrained (dict, optional): Path Dict to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            assert isinstance(pretrained, dict)
            logger = get_root_logger()
            print_log(f'load weight from: {pretrained}', 
                      logger=logger)
            self.encoder.init_weights(pretrained['encoder_pretrained'])
            self.decoder.init_weights(pretrained['decoder_pretrained'])

    async def aforward_test(self, 
                            img_metas, 
                            img, 
                            img_mask,
                            cap,
                            cap_mask,
                            **kwargs):
        pass

    def forward_test(self, 
                     img_metas, 
                     img,
                     img_mask, 
                     cap,
                     cap_mask,
                     **kwargs):
        """Forward Test

        """
        return self.simple_test(img_metas, 
                                img, 
                                img_mask,
                                cap,
                                cap_mask,
                                **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, 
                img_metas, 
                img, 
                img_mask,
                cap,
                cap_mask,
                return_loss=True, 
                **kwargs): 
        """Forward
        Calls either 
        :func:`forward_train` or 
        :func:`forward_test` 
        depending on whether ``return_loss`` is ``True``.

        """
        if return_loss:
           return self.forward_train(img_metas,
                                     img,
                                     img_mask,
                                     cap,
                                     cap_mask, **kwargs)
        else: 
            return self.forward_test(img_metas,
                                     img,
                                     img_mask,
                                     cap,
                                     cap_mask, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses, _ = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer): 
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses, _ = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def generate_caption(self,
                         img_metas,
                         img,
                         img_mask,
                         cap,
                         cap_mask,
                         decoding_cfg,
                         **kwargs):
        """Generate caption result

        """
        if decoding_cfg.type == "greedy":
            return self.generate_greedy_caption(img_metas,
                                                img,
                                                img_mask,
                                                cap,
                                                cap_mask,
                                                **kwargs)
        elif decoding_cfg.type == "topktopp":
            return self.generate_topktopp_caption(img_metas,
                                                  img,
                                                  img_mask,
                                                  cap,
                                                  cap_mask,
                                                  decoding_cfg.topk,
                                                  decoding_cfg.topp,
                                                  **kwargs)

    @abstractmethod
    def generate_greedy_caption(self,
                                img_metas,
                                img,
                                img_mask,
                                cap,
                                cap_mask,
                                **kwargs):
        """Generate caption result by greedy search

        """
        pass

    @abstractmethod
    def generate_topktopp_caption(self,
                                  img_metas,
                                  img,
                                  img_mask,
                                  cap,
                                  cap_mask,
                                  topk=5,
                                  topp=0.8,
                                  **kwargs):
        """Generate top K, top P decoding result

        """
        pass


class FFNN(nn.Module):
    """ FFNN

    Classify token from caption feature.
    """

    def __init__(self, input_dim, hidden_dim:list, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        #h = [hidden_dim] * (num_layers - 1)
        h = hidden_dim * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
