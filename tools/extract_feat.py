"""
Luke
"""
import os.path as osp
import glob
import tqdm

import numpy as np
import mmcv

from mmcap.apis.inference import init_caption
from mmcap.apis.extract import extract_encoder_feat
from mmcap.datasets.tokenizers.tokenization_kobert import KoBertTokenizer


def extract_feat(model, tokenizer, src_dir:str, dst_dir:str):
    img_pathes = glob.glob('{}*.jpg'.format(src_dir))

    for img_path in tqdm.tqdm(img_pathes, total=len(img_pathes)):
        result = extract_encoder_feat(model, tokenizer, img_path)
        img_id = img_path.split('/')[-1].split('.')[0]

        feat_dst_path = osp.join(dst_dir, '{}_feat.pkl'.format(img_id))
        mask_dst_path = osp.join(dst_dir, '{}_mask.pkl'.format(img_id))
        pos_dst_path = osp.join(dst_dir, '{}_pos.pkl'.format(img_id))
        # dump result
        with open(feat_dst_path, 'wb') as f:
            mmcv.dump(result[0], f, file_format='pkl')
        with open(mask_dst_path, 'wb') as f:
            mmcv.dump(result[1], f, file_format='pkl')
        with open(pos_dst_path, 'wb') as f:
            mmcv.dump(result[2], f, file_format='pkl')


def main():
    dir_info = [{'mode': 'train',
                 'src': 'data/coco/train2017/',
                 'dst': 'data/coco/features_train2017/'},
                {'mode': 'val',
                 'src': 'data/coco/val2017/',
                 'dst': 'data/coco/features_val2017/'}]

    config = 'configs/image_captioning/cbnet_transformer_test_nondist.py'
    
    model = init_caption(config)

    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    for dir_dic in dir_info:
        print('\nProcessing {}...'.format(dir_dic['mode']))
        src_dir = dir_dic['src']
        dst_dir = dir_dic['dst']
        extract_feat(model, tokenizer, src_dir, dst_dir)


if __name__ == "__main__":
    main()
