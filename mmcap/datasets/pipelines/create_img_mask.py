import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class CreateImgMask(object):
    """ Pad Image, and Create Image Mask for Given Image.

    """
    def __init__(self, max_dim:int):
        self.max_dim = max_dim

    def __call__(self, results:dict):
        img = results['img']
        img = img[None]

        if img[0].ndim == 3:
            max_size = [3, self.max_dim, self.max_dim]
            shape = [1] + max_size
            _, c, h, w = shape
            dtype = img.dtype
            device = img.device
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            mask = torch.ones((1, h, w), dtype=torch.bool, device=device)

            for im, pad_img, m in zip(img, tensor, mask):
                pad_img[: im.shape[0], : im.shape[1], : im.shape[2]].copy_(im)
                m[: im.shape[1], :im.shape[2]] = False
        
        else:
            raise ValueError('not supported')

        results['img'] = pad_img
        results['img_mask'] = mask
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
