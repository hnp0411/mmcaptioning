# optimizer
# optimizer = dict(type='AdamW',
#                  weight_decay = 1e-4)
optimizer_config = dict(grad_clip=None)

# lr
lr_dict = dict(lr=1e-4,
               lr_backbone=1e-5)
weight_decay = 1e-4

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-5,
    step=[20, 29])
total_epochs = 30
