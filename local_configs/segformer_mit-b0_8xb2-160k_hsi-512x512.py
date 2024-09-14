_base_ = [
    './_base_/models/segformer_mitdlp-b0.py', './_base_/datasets/hsi_facade.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_epoch.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
#checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa


model = dict(
    data_preprocessor=data_preprocessor,

    decode_head=dict(num_classes=42, # 44 ' = 40 clean classes
                     #n_channels=[64, 128, 320, 512],
                     loss_decode=[
                    #dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    dict(type='FocalLoss', loss_name='loss_focal', loss_weight=3.0, alpha=0.25, gamma=3.0),
                    #dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, use_sigmoid=False)

                    ])
                )


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    #dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=15),
    # Use a cosine learning rate at [100, 900) iterations
    #dict(
    #    type='CosineAnnealingLR',
    #    T_max=200,
    #    by_epoch=True,
    #    begin=15,
    #    end=200),
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=15),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=15,
        end=200,
        by_epoch=True,
    )
]
train_dataloader = dict(batch_size=20, num_workers=12)
val_dataloader = dict(batch_size=1, num_workers=12)
test_dataloader = dict(batch_size=4, num_workers=12)
