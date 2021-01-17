import os.path as osp
import copy
import torch

import mmcv

from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class TextGenerateRunner(EpochBasedRunner):
    def set_gen_iter(self, gen_batch):
        self.gen_batch = gen_batch

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_decoding_cfg(self, decoding_cfg):
        self.decoding_cfg = decoding_cfg

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
            # TODO : assert not distributed
            if self.inner_iter % self.gen_batch == 0:
                self.generate_sample(data_batch)

        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def generate_sample(self, data_batch):

        generate_sample = dict()
        start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        caption_template = torch.zeros((1, 129), dtype=torch.long)
        mask_template = torch.ones((1, 129), dtype=torch.bool)
        caption_template[:,0] = start_token
        mask_template[:,0] = False

        for key in data_batch:
            if not key in ["cap", "cap_mask"]:
                generate_sample[key] = copy.deepcopy(data_batch[key]._data[0][:1])
            elif key == "cap": 
                generate_sample[key] = caption_template
            elif key == "cap_mask":
                generate_sample[key] = mask_template
            else:
                pass

            if key != "img_metas":
                generate_sample[key] = generate_sample[key].to(device='cuda:0')

        generate_sample['decoding_cfg'] = self.decoding_cfg
        
        results = self.model.module.generate_caption(**generate_sample)
        caption = self.tokenizer.decode(results[0].tolist())#, skip_special_tokens=True)
        print('-> Sample Filename : {}'.format(generate_sample['img_metas'][0]['filename']))
        gt = data_batch['cap']._data[0][:1]
        raw_gt = generate_sample['img_metas'][0]['raw_cap']

        print('-> Sample GroundTruth Tensor : {}'.format(gt))
        print('-> Sample GroundTruth Caption : {}'.format(raw_gt))
        print('-> Sample Generated Tensor : {}'.format(results))
        print('-> Sample Generated Caption : {}\n'.format(caption))
