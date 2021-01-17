import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class EncodeCaption(object):
    """ Encode given captions

    """
    def __init__(self, 
                 caption_max_length:int,
                 padding:str,
                 truncation:bool):
        self.caption_max_length = caption_max_length + 1
        self.padding = padding
        self.truncation = truncation

    def _load_tokenizer(self, results:dict):
        self.tokenizer = results['cap_info']['tokenizer']
        self.tokenizer_type = results['cap_info']['tokenizer_cfg']['type']

    def _encode(self, raw_cap):
        if 'Wrapper' in self.tokenizer_type:
            return self._encode_huggingface_tokenizer(raw_cap)

    def _encode_huggingface_tokenizer(self, raw_cap):
        caption_encoded = self.tokenizer.encode_plus(
            raw_cap, 
            max_length=self.caption_max_length,
            padding=self.padding,
            truncation=self.truncation)

        np_caption = np.array(caption_encoded['input_ids'])
        np_cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return np_caption, np_cap_mask

    def __call__(self, results:dict):
        """Encode Captions for train and evaluation.
        """
        self._load_tokenizer(results)

        raw_cap = results['raw_cap']
        np_caption, np_cap_mask = self._encode(raw_cap)

        results['cap'] = np_caption
        results['cap_mask'] = np_cap_mask
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class EncodeEmptyCaption(object):
    """ Encode empty captions for inference

    """
    def __init__(self, 
                 caption_max_length:int,
                 padding:str,
                 truncation:bool):
        self.caption_max_length = caption_max_length + 1
        self.padding = padding
        self.truncation = truncation

    def _load_tokenizer(self, results:dict):
        self.tokenizer = results['cap_info']['tokenizer']
        self.tokenizer_type = results['cap_info']['tokenizer_cfg']['type']

    def _encode(self):
        if 'Wrapper' in self.tokenizer_type:
            return self._encode_huggingface_tokenizer()

    def _encode_huggingface_tokenizer(self):
        start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        caption_template = torch.zeros((1, self.caption_max_length), dtype=torch.long)
        mask_template = torch.ones((1, self.caption_max_length), dtype=torch.bool)
    
        caption_template[:, 0] = start_token
        mask_template[:, 0] = False
        return caption_template, mask_template

    def __call__(self, results:dict):
        """Encode Empty Caption for predict and api.
        """
        self._load_tokenizer(results)

        caption, cap_mask = self._encode()

        results['cap'] = caption
        results['cap_mask'] = cap_mask

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
