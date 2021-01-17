"""
Luke

KoBertTokenizer Wrapper for mmcaptioning
"""

from mmcap.utils import KoBertTokenizer

from . import TOKENIZERS


@TOKENIZERS.register_module()
class KoBertTokenizerWrapper():
    """KoBert Tokenizer Wrapper

    """

    def __init__(self,
                 pretrained=None):

        self.tokenizer = KoBertTokenizer.from_pretrained(pretrained)
