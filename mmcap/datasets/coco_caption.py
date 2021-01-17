import os.path as osp
import warnings

import mmcv
from mmcv.parallel import DataContainer as DC
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from mmcap.tokenizers import build_tokenizer
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CocoCaption(Dataset):
    """Coco Caption dataset for Image captioning.

    Args:
        ann_file (str): COCO Caption Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 tokenizer,
                 data_root=None,
                 img_prefix='',
                 test_mode=False):

        self.coco = COCO(ann_file)
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.tokenizer_cfg = tokenizer
        self.tokenizer = build_tokenizer(tokenizer)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # load image and caption
        self.cap_infos = self.load_caps()
        self.img_infos = self.load_imgs()

        if not self.test_mode: 
            # set group flag for the sampler if train mode
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of captions.

        """
        return len(self.cap_infos)

    def load_caps(self):
        """Load COCO caption list from ann_file

        used only train now
        eval metric 개발 후에 코드 수정 필요함

        """
        cap_infos = list()
        for cap_id, cap_dic in self.coco.anns.items():
            cap_infos.append(cap_dic)
        return cap_infos

    def load_imgs(self):
        """Load COCO imgs dict from ann_file

        """
        return self.coco.imgs

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        flag = np.zeros(len(self), dtype=np.uint8)
        for ind in range(len(self)):
            img_id = self.cap_infos[ind]['image_id']
            img_info = self.img_infos[img_id]
            if img_info['width'] / img_info['height'] > 1:
                flag[ind] = 1
        self.flag = flag

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of Caption data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        cap_info = self.cap_infos[idx]
        image_id = cap_info['image_id']
        cap_id = cap_info['id']
        caption = cap_info['caption']
        img_info = self.img_infos[image_id]
        img_info['filename'] = img_info['file_name']
        cap_info['tokenizer'] = self.tokenizer
        cap_info['tokenizer_cfg'] = self.tokenizer_cfg

        results = dict(img_info=img_info,
                       cap_info=cap_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of Caption data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """
        cap_info = self.cap_infos[idx]
        image_id = cap_info['image_id']
        cap_id = cap_info['id']
        caption = cap_info['caption']
        img_info = self.img_infos[image_id]
        img_info['filename'] = img_info['file_name']
        cap_info['tokenizer'] = self.tokenizer
        cap_info['tokenizer_cfg'] = self.tokenizer_cfg

        results = dict(img_info=img_info, 
                       cap_info=cap_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    # image captioning eval : return validation loss
    def evaluate(self,
                 results,
                 metric='loss',
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['loss']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = dict()

        total_loss = 0
        if metric == 'loss':
            for result in results:
                total_loss += result['loss']
        eval_results['loss'] = total_loss / len(results)

        return eval_results
