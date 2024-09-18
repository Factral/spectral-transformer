
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadSpectralImageFromNpyFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', prob=0.5, degree=35),
    dict(type='PackSegSpectralInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadSpectralImageFromNpyFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegSpectralInputs')
]

DATA_ROOT = './data/LIB-HSI-fixed'


train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='HSIFacade',
        data_root=DATA_ROOT,
        data_prefix=dict(
            img_path='train/rgb',
            seg_map_path='train/labels',
            spectral_path='train/reflectance_cubes'),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIFacade',
        data_root=DATA_ROOT,
        data_prefix=dict(
            img_path='test/rgb',
            seg_map_path='test/labels',
            spectral_path='test/reflectance_cubes'),
        pipeline=test_pipeline))


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIFacade',
        data_root=DATA_ROOT,
        data_prefix=dict(
            img_path='test/rgb',
            seg_map_path='test/labels',
            spectral_path='test/reflectance_cubes'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator