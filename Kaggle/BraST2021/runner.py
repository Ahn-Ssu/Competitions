import os

# def replace_module(modules:nn.Module, target, source):
#         for name, child in modules.named_children():
#             if isinstance(child, target):
#                 modules._modules[name] = source()
#             # elif isinstance(child, nn.Sequential):
#             else: 
#                 replace_module(child, target, source)


if __name__ == '__main__':

    from easydict import EasyDict
    from torch.utils.data import DataLoader
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

    import torch.nn as nn
    

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

    from monai.networks import nets
    from monai.networks.blocks import MemoryEfficientSwish as MEMSWISH
    from data_loader import MRI_dataset
    from lighting import LightningRunner
    from kfold_pl_data import KFold_pl_DataModule


    args = EasyDict()

    args.batch_size = 2
    args.epochs = 300
    args.init_lr = 4e-3
    args.weight_decay = 0.05

    args.seed = 41

    seed_everything(args.seed)

    # 240, 240, 155
    # Swin에서 사용했던 aug
    # 128 cube random cropped 
    # axis mirror flip p=0.5
    # per channel intensity shift (-0.1, 0.1)
    # random scale intensity (0.9, 1.1)
    train_transform = Compose(
            [
                # load 4 Nifti images and stack them together
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # the following from this is augmentation
                # CropForegroundd(keys=["image","label"], source_key="image", k_divisible=[240, 240, 155]),
                # RandSpatialCropd(keys=["image", "label"], roi_size=[128,128,128], random_size=False),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    val_transform = Compose(
            [
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
    
    num_split = 5
    for k_idx in range(num_split):

        pl_data = KFold_pl_DataModule(data_dir='/root/Competitions/Kaggle/BraST2021/data/BraTS2021_Training_Data',
            ext='nii.gz',
            k_idx=k_idx,
            num_split=num_split,
            split_seed=args.seed,
            batch_size=args.batch_size,
            num_workers=4,
            persistent_workers=False,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform
            )

        model = nets.BasicUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            # features=
            # features=(64,128,256,512,1024,64),
            # act=("ReLU",{'inplace':True}),
            # act=("MEMSWISH", {})
            # dropout=0.05
        )

        # model = nets.SwinUNETR(
        #     img_size=[128,128,128],
        #     in_channels=4,
        #     out_channels=3
        # )

        print(model)
        # exit()

        pl_runner = LightningRunner(model, args)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        lr_finder = LearningRateFinder(num_training_steps=8000)
        checkpoint_callback = ModelCheckpoint(
            monitor='avg_dice',
            filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{avg_dice:.4f}',
            mode='max'
        )

        logger = TensorBoardLogger(
            save_dir='.',
            version=f'[{k_idx+1}Fold]_BasicUNet, ReLU, lr 3e-4, whole img(240x240x155) no foreC, lr find'
            )
    
        trainer = Trainer(
            max_epochs=args.epochs,
            devices=4,
            accelerator='gpu',
            precision=16,
            strategy=DDPStrategy(find_unused_parameters=False),
            callbacks=[lr_monitor, checkpoint_callback, lr_finder],
            check_val_every_n_epoch=10,
            logger=logger
            )

        trainer.fit(
            model= pl_runner,
            datamodule=pl_data
        )
