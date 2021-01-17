_base_ = [
    '../_base_/models/res2net101_transformer_koelectra.py',
    '../_base_/tokenizers/koelectra_tokenizer.py',
    '../_base_/datasets/coco_caption_nondist.py',
    '../_base_/schedules/schedule_30e.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/fp16.py'
]
