"""
Luke
"""
import argparse
import glob
import os.path as osp

from mmcap.apis.inference import init_caption, generate_caption
from mmcap.tokenizers import build_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('img', help='input image path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    checkpoint_dir = 'checkpoints/res2net101_transformer_koelectra_30e_fp16_coco_nondist/'
    checkpoint_path = osp.join(checkpoint_dir, 'latest.pth')
    config = 'configs/resnet_transformer/res2net101_transformer_koelectra_30e_fp16_coco_nondist.py'
   
    # Initialize Caption Model
    model = init_caption(config, device='cuda:1', checkpoint=checkpoint_path)

    # Initialize Tokenizer
    tokenizer_cfg = model.cfg.tokenizer
    tokenizer = build_tokenizer(tokenizer_cfg)

    # Get Generated Caption Result
    caption = generate_caption(model, 
                               tokenizer, 
                               tokenizer_cfg, 
                               args.img)

    print('\nGenerated Caption : {}\n'.format(caption))


if __name__ == "__main__":
    main()
