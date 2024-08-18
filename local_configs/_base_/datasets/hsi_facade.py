train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=20),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='HSIFacade',
        data_root='data/LIB-HSI',
        data_prefix=dict(
            img_path='train/rgb', seg_map_path='train/labels'),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIFacade',
        data_root='data/LIB-HSI',
        data_prefix=dict(
            img_path='validation/rgb',
            seg_map_path='validation/labels'),
        pipeline=test_pipeline))


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIFacade',
        data_root='data/LIB-HSI',
        data_prefix=dict(
            img_path='test/rgb',
            seg_map_path='test/labels'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator