# model settings

model = dict(
    type='ResNetTransformer',
    pretrained=dict(
        encoder_pretrained=dict(
            backbone_pretrained='open-mmlab://res2net101_v1d_26w_4s'),
        decoder_pretrained=dict(
            transformer_pretrained='Default')
    ),
    ffnn_hidden_dims=[512],
    ffnn_num_layers=3,
    encoder=dict(
        type='Res2NetEncoder',
        backbone_feats=2048,
        pos_feats=128,
        pos_temperature=10000,
        pos_norm=True,
        backbone=dict(
            type='Res2Net',
            depth=101,
            scales=4,
            base_width=26,
            strides=(1, 2, 2, 1)
        )
    ),
    decoder=dict(
            type='TextDecoder',
            hidden_dim=256,
            pad_token_id=0,
            max_position_embeddings=128,
            layer_norm_eps=1e-12,
            dropout=0.1,
            vocab_size=35000,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            nheads=8,
            pre_norm=True
        )
    )

train_cfg = dict(
    decoding_cfg=dict(
        type='topktopp',
        topk=1,
        topp=1
    )
)
test_cfg = dict(
    decoding_cfg=dict(
        type='topktopp',
        topk=1,
        topp=1
    )
)
