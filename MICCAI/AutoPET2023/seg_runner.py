
def run():
    import os
    from datetime import date, datetime, timezone, timedelta

    import numpy as np
    import pandas as pd 
    from easydict import EasyDict

    import torch.storage

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder
    from pytorch_lightning.loggers import TensorBoardLogger

    from monai.transforms import (
                            Compose,
                            OneOf,

                            LoadImaged,
                            EnsureTyped,
                            ScaleIntensityRanged,
                            Orientationd,
                            CropForegroundd, 
                            RandCropByPosNegLabeld,
                            RandSpatialCropd,

                            RandFlipd,
                            RandRotated,
                            RandZoomd,

                            RandShiftIntensityd,
                            RandScaleIntensityd,
                            RandAdjustContrastd,
                            RandGaussianNoised,
                            RandGaussianSmoothd,
                            RandGaussianSharpend,
                            HistogramNormalized,

                            RandCoarseDropoutd,
                            RandCoarseShuffled
                        )
    from monai.data import set_track_meta

    from dataloader import KFold_pl_DataModule
    from model import unet_baseline, late_fusion, tail_fusion
    from seg_pl import Segmentation_network

    from monai.networks.nets import BasicUNet, SwinUNETR

    args = EasyDict()

    args.img_size = 128
    args.batch_size = 1
    args.epoch = 200
    args.init_lr = 1e-4
    args.weight_decay = 0.05

    args.seed = 41
    seed.seed_everything(args.seed)


    all_key = ['ct','pet','label']

    test_transform = Compose([
        LoadImaged(keys=all_key, ensure_channel_first=True),
        EnsureTyped(keys=all_key, track_meta=False),
        Orientationd(keys=all_key, axcodes='RAS'),
        ScaleIntensityRanged(keys='ct',
                                 a_min=-1000, a_max=1000,
                                 b_min=0, b_max=1, clip=True),
        ScaleIntensityRanged(keys='pet',
                                a_min=0, a_max=40,
                                b_min=0, b_max=1, clip=True),

        CropForegroundd(keys=all_key, source_key='pet'), # source_key 'ct' or 'pet'
    ]
    )

    train_transform = Compose([
            LoadImaged(keys=all_key, ensure_channel_first=True),
            EnsureTyped(keys=all_key, track_meta=False), # for training track_meta=False, monai.data.set_track_meta(false)
            Orientationd(keys=all_key, axcodes='RAS'),
            ScaleIntensityRanged(keys='ct',
                                 a_min=-1000, a_max=1000,
                                 b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys='pet',
                                 a_min=0, a_max=40,
                                 b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=all_key, source_key='pet'), # source_key 'ct' or 'pet'
            OneOf([
                RandCropByPosNegLabeld(keys=all_key, label_key='label', 
                                       spatial_size=(args.img_size,args.img_size,args.img_size), 
                                       pos=1, neg=0.2, num_samples=1,
                                       image_key='pet',
                                       image_threshold=0), # 흑색종일때 label에 따라서 잘 되는지 확인해야함 
                RandSpatialCropd(keys=all_key, roi_size=[args.img_size,args.img_size,args.img_size], random_size=False
                                 )],
                weights=[0.8, 0.2],
                ),
            # spatial
            # RandFlipd(keys=all_key, prob=0.5, spatial_axis=0), 
            # RandFlipd(keys=all_key, prob=0.1, spatial_axis=1),
            # RandFlipd(keys=all_key, prob=0.1, spatial_axis=2),
            # # RandRotated(keys=all_key, range_x=20, range_y=5, range_z=5, prob=0.2),
            # # RandZoomd(keys=all_key, prob=0.2),

            # # intensity
            # RandShiftIntensityd(keys=['pet','ct'], offsets=0.01, prob=0.2),
            # RandScaleIntensityd(keys=['pet','ct'], prob=0.2, factors=0.1),
            # RandAdjustContrastd(keys=['pet','ct'], prob=0.2),
            # OneOf([
            #     RandGaussianNoised(keys=['pet','ct'], prob=0.2),
            #     RandGaussianSmoothd(keys=['pet','ct'], prob=0.2),
            #     RandGaussianSharpend(keys=['pet','ct'], prob=0.2),
            # ]),
            
            # OneOf([
            #     RandCoarseDropoutd(keys=['pet','ct'], prob=0.2, holes=10, spatial_size=10),
            #     RandCoarseShuffled(keys=['pet','ct'], prob=0.2, holes=10, spatial_size=10)
            # ]
            # )
        ]) # ㄱㅣㅈㅗㄴ [-100, 400] - [0, 30] || ㅈㅣㄱㅡㅁ ㄷㅗㄹㅇㅏㄱㅏㄴㅡㄴ ㄱㅓ -100 500, 0 100

    # set_track_meta(False)

    num_split = 5
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):

        pl_dataloder = KFold_pl_DataModule(
                            data_dir='/root/Competitions/MICCAI/AutoPET2023/data/train',
                            k_idx=fold_idx,
                            num_split=num_split,
                            split_seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=8,
                            pin_memory=False,
                            persistent_workers=True,
                            train_transform=train_transform,
                            val_transform=test_transform
                        )

        # model = unet_baseline.UNet(
        #                     input_dim=2,
        #                     out_dim=2,
        #                     hidden_dims=[32,64,128,256,512], # 16 32 32 64 128 is default setting of Monai
        #                     spatial_dim=3,
        #                     dropout_p=0.
        #                 )

        model = late_fusion.UNet_lateF(
                            input_dim=3,
                            out_dim=2,
                            hidden_dims=[16,32,64,128,256], # 16 32 32 64 128 is default setting of Monai
                            spatial_dim=3,
                            dropout_p=0.,
                            use_MS=False
                        )
        
        # model = tail_fusion.UNet_tailF(
        #                     input_dim=3,
        #                     out_dim=2,
        #                     hidden_dims=[16,32,64,128,256], # 16 32 32 64 128 is default setting of Monai
        #                     spatial_dim=3,
        #                     dropout_p=0.,
        #                     use_MS=True
        #     )
        
        # model = SwinUNETR(
        #     img_size=128,
        #     in_channels=2,
        #     out_channels=2,
        #     feature_size=24,
        #     spatial_dims=3
        # )
        
        print(model)
        
        pl_runner = Segmentation_network(network=model, args=args)#.load_from_checkpoint('/root/Competitions/MICCAI/AutoPET2023/lightning_logs/IntensityRange/2023-06-28/CT=(-100, 400), PET=(0, 40) || UNet_lateF(16,256) w He - GPU devices[2,3]/checkpoints/UNet_lateF-epoch=182-train_loss=0.3444-val_dice=0.6879.ckpt', network=model, args=args)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_dice',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{val_dice:.4f}',
                                    mode='max',
                                    # save_top_k=1,
                                    save_last=True
                                )
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            # version='LEARNING CHECK',
                            # version=f'IntensityRange/{_day}/[{fold_idx+1} Fold] Aug(offset=0.01), SUV(0,40) || CFG == -m UNet_lateF(16-256) AdamP -lr {args.init_lr} -img_sz {args.img_size}] bz {args.batch_size} x 2GPU(2,3)'
                            version=f'IntensityRange/{_day}/CT=(-1000, 1000), PET=(0, 40) || UNet_lateF(16,256) w He+MS - GPU devices[2,3]'
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[2,3],
                    accelerator='gpu',
                    # precision='16-mixed',
                    strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback],
                    # check_val_every_n_epoch=2,
                    check_val_every_n_epoch=3,
                    # log_every_n_steps=1,
                    logger=logger,
                    # auto_lr_find=True
                    # accumulate_grad_batches=2
                    # profiler='simple', #advanced
                )
        

        
        trainer.fit(
                model= pl_runner,
                datamodule= pl_dataloder,
                # If we wanna keep training use this code line 
                # 근데 잘 안되요잉
                # ckpt_path='/root/Competitions/MICCAI/AutoPET2023/lightning_logs/IntensityRange/2023-06-28/CT=(-100, 400), PET=(0, 40) || UNet_lateF(16,256) w He - GPU devices[2,3]/checkpoints/UNet_lateF-epoch=182-train_loss=0.3444-val_dice=0.6879.ckpt'
            )
        
        break;

    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()