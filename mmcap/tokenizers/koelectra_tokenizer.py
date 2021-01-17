"""
Luke

KoELECTRA Tokenizer Wrapper for mmcaptioning
"""

from transformers import ElectraTokenizer

from . import TOKENIZERS


@TOKENIZERS.register_module()
class KoElectraTokenizerWrapper():
    """KoELECTRA Tokenizer Wrapper

    """

    def __init__(self,
                 pretrained=None):

        self.tokenizer = ElectraTokenizer.from_pretrained(pretrained)
