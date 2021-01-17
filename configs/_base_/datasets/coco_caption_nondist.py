dataset_type = 'CocoCaption'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_dim = 299

train_pipeline=[
    # preprocess img
    dict(type='LoadPILImage'),
    dict(type='RandomRotation', angles=[0, 90, 180, 270]),
    dict(type='UnderMax', max_dim=max_dim),
    dict(type='ColorJitter',
         brightness=[0.5, 1.3],
         contrast=[0.8, 1.5],
         saturation=[0.2, 1.5]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='ToTensorCaptioning', keys=['img']),
    dict(type='NormalizeCaptioning'),
    # create img_mask
    dict(type='CreateImgMask', max_dim=max_dim),
    # preprocess caption
    dict(type='LoadCaption'),
    dict(type='EncodeCaption',
         caption_max_length=128,
         padding='max_length',
         truncation=True),
    # from np array to DataContainer
    dict(type='DefaultFormatBundleCaptioning'),
    dict(type='Collect',
         keys=['img', 'img_mask', 'cap', 'cap_mask'],
         meta_keys=['filename', 'raw_cap'])
]

test_pipeline=[
    # preprocess img
    dict(type='LoadPILImage'),
    dict(type='UnderMax', max_dim=max_dim),
    dict(type='ToTensorCaptioning', keys = ['img']),
    dict(type='NormalizeCaptioning'),
    # create img_mask
    dict(type='CreateImgMask', max_dim=max_dim),
    # preprocess caption
    dict(type='LoadCaption'),
    dict(type='EncodeCaption',
         caption_max_length=128,
         padding='max_length',
         truncation=True),
    # from np array to DataContainer
    dict(type='DefaultFormatBundleCaptioning'),
    dict(type='Collect',
         keys=['img', 'img_mask', 'cap', 'cap_mask'],
         meta_keys=['filename', 'raw_cap'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8, # NGPUS(2) * WORKERS_PER_GPU = N_CPU_CORES(16)
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/captions_kor_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/captions_kor_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/captions_kor_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
)

evaluation = dict(interval=1, metric='loss')
