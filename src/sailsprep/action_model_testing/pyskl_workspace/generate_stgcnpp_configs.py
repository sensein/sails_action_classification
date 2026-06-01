# save as: generate_stgcnpp_configs.py
import os

template = """
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes={num_classes}, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '{ann_file}'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/{model_name}_{dataset_name}/{feat}'
"""

datasets = {
    'rmm': {
        'ann_file': '/home/aparnabg/orcd/pool/pyskl_workspace/data/rmm_pyskl.pkl',
        'num_classes': 4
    },
    'loco': {
        'ann_file': '/home/aparnabg/orcd/pool/pyskl_workspace/data/loco_pyskl.pkl',
        'num_classes': 5
    }
}

feats = ['j', 'b', 'jm', 'bm']

for ds_name, ds_info in datasets.items():
    out_dir = f'configs/custom/stgcnpp_{ds_name}'
    os.makedirs(out_dir, exist_ok=True)
    for feat in feats:
        config = template.format(
            num_classes=ds_info['num_classes'],
            ann_file=ds_info['ann_file'],
            feat=feat,
            model_name='stgcnpp',
            dataset_name=ds_name
        )
        with open(os.path.join(out_dir, f'{feat}.py'), 'w') as f:
            f.write(config.strip())
        print(f'Created {out_dir}/{feat}.py')