# model settings
from mmseg.models.builder import BACKBONES, SEGMENTORS

norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='./mit_b4.pth',
    backbone=dict(
                type=BACKBONES.get('mit_b4_vpt'),
                img_size=512,
                prompt_cfg='deep',),
     decode_head=dict(
                type='SegFormerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=768),
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
