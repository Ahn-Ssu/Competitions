
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
                            Activations,
                            Activationsd,
                            AsDiscrete,
                            AsDiscreted,
                            ConvertToMultiChannelBasedOnBratsClassesd,
                            Compose,
                            Invertd,
                            LoadImaged,
                            MapTransform,
                            NormalizeIntensityd,
                            Orientationd,
                            RandFlipd,
                            RandScaleIntensityd,
                            RandShiftIntensityd,
                            RandSpatialCropd,
                            CropForegroundd,
                            RandAffined,
                            Resized,
                            Spacingd,
                            EnsureTyped,
                            EnsureChannelFirstd,
                        )

    from dataloader import KFold_pl_DataModule
    from model import unet_baseline
    from lightning import LightningRunner

    from monai.networks.nets import UNet

    args = EasyDict()

    args.img_size = 144
    args.batch_size = 1
    args.epoch = 80
    args.init_lr = 1e-4
    args.weight_decay = 0.05

    args.seed = 41
    seed.seed_everything(args.seed)


    train_transform = Compose(
                [
                    # load 4 Nifti images and stack them together
                    EnsureChannelFirstd(keys=["image", "label"]),
                    EnsureTyped(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    # Spacingd(
                    #     keys=["image", "label"],
                    #     pixdim=(1.0, 1.0, 1.0),
                    #     mode=("bilinear", "nearest"),
                    # ),
                    # the following from this is augmentation
                    # CropForegroundd(keys=["image","label"], source_key="image", k_divisible=[192, 192, 192]),
                    RandSpatialCropd(keys=["image", "label"], roi_size=[args.img_size,args.img_size,args.img_size], random_size=False),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                ]
            )
    test_transform = Compose(
                [
                    EnsureChannelFirstd(keys=["image", "label"]),
                    EnsureTyped(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    # Spacingd(
                    #     keys=["image", "label"],
                    #     pixdim=(1.0, 1.0, 1.0),
                    #     mode=("bilinear", "nearest"),
                    # ),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            )


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

        model = unet_baseline.UNet(
                            input_dim=2,
                            out_dim=1,
                            hidden_dims=[16,32,32,64,128], # 16 32 32 64 128 is default setting of Monai
                            spatial_dim=3,
                            dropout_p=0.
                        )
        pl_runner = LightningRunner(network=model, args=args)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='avg_f1',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{avg_f1:.4f}',
                                    mode='max'
                                )
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            version='LEARNING CHECK',
                            # version=f'{_day}/[{fold_idx+1} Fold] REPRODUCE -m convnext_large, -d P, -t GV -opt AdamP || lr=[{args.init_lr}] img=[{args.img_size}] bz=[{args.batch_size}] 2gpu'
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[0,1,2,3],
                    accelerator='gpu',
                    precision='16-mixed',
                    # strategy=DDPStrategy(find_unused_parameters=False),
                    callbacks=[lr_monitor, checkpoint_callback],
                    # check_val_every_n_epoch=2,
                    check_val_every_n_epoch=5,
                    # log_every_n_steps=1,
                    logger=logger,
                    # auto_lr_find=True
                    # accumulate_grad_batches=2
                )
        
        trainer.fit(
                model= pl_runner,
                datamodule= pl_dataloder
            )

    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()