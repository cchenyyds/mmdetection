_base_ = [
    '../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
default_scope = 'mmdet'
model = dict(

    data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=None,
            std=None,
            #bgr_to_rgb=True,
            pad_mask=True,
            bgr_to_rgb=False,
            pad_size_divisor=224),

    #pretrained='pretrained/alt_gvt_small.pth',
    #data_processor=dict(
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32)
    #),

    backbone=dict(
        _delete_=True,
        type='Dual_s',
        #out_indices=(0, 1, 2, 3),
        #style='pytorch'
    ),

    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256,),

    roi_head=dict(
         bbox_head=dict(num_classes=1),
         mask_head=dict(num_classes=1),),
    test_cfg = dict(rcnn=dict(
        score_thr=0.001,))
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='Mytransformer'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='RandomChoiceResize',
                      scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],

                      keep_ratio=True)
             ],
             [
                 dict(type='RandomChoiceResize',
                      scales=[(400, 1333), (500, 1333), (600, 1333)],

                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='RandomChoiceResize',
                      scales=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],


                      keep_ratio=True)
             ]
         ]),
    dict(
        type='PackDetInputs',
        # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor','rgb_path',"seq_id")),
    #dict(type='PackDetInputs', meta_keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]


test_pipeline = [  # Testing data processing pipeline
    dict(type='Mytransformer'),  # First pipeline to load images from file path
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # Pipeline that resizes the images
    dict(
        type='PackDetInputs',  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor','rgb_path',"seq_id")),
]

classes = ('drone')
metainfo = dict(classes = classes)

train_dataloader = dict(batch_size=1,num_workers=2,sampler=dict(type='DefaultSampler',shuffle=True),dataset = dict(type="RGBEventDataset",pipeline=train_pipeline,metainfo = metainfo),batch_sampler=dict(type='AspectRatioBatchSampler'))

val_dataloader = dict(  # Validation dataloader config
    batch_size=1,  # Batch size of a single GPU. If batch-size > 1, the extra padding area may influence the performance.
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    drop_last=False,  # Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),  # not shuffle during validation and testing
    dataset=dict(
        type="RGBEventDataset",
        metainfo = metainfo,
        test_mode=True,  # Turn on the test mode of the dataset to avoid filtering annotations or images
        pipeline=test_pipeline))



test_dataloader = val_dataloader
train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=15,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.

test_cfg = dict(type='TestLoop')  # The testing loop type


val_cfg = dict(type='ValLoop')  # The validation loop type
optim_wrapper = dict(  # Optimizer wrapper config

    type='AmpOptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict( type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
                   paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),'norm': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=35, norm_type=2)
    )

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    #ann_file=data_root + 'annotations/instances_val2017.json',  # Annotation file path
    metric=['bbox', 'segm'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    #backend_args=backend_args
)
test_evaluator = val_evaluator  # Testing evaluator config




# optimizer
optimizer = None
optimizer_config = None

# learning policy
lr_config = dict(step=[18, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
total_epochs = 24
fp16 = dict(loss_scale=512.)
