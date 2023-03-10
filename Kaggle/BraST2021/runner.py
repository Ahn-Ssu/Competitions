import os

if __name__ == '__main__':

    from easydict import EasyDict
    from torch.utils.data import DataLoader
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
    

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
                        Spacingd,
                        EnsureTyped,
                        EnsureChannelFirstd,
                    )

    from monai.networks import nets
    from data_loader import MRI_dataset
    from lighting import LightningRunner
    from models.model import DeepSEED


    args = EasyDict()

    args.batch_size = 2
    args.epochs = 400
    args.init_lr = 0.001

    args.seed = 41

    seed_everything(args.seed)

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
                RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
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
    train_data = MRI_dataset('./data/train_originWhole', 'nii.gz', train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    val_data = MRI_dataset('/root/Competitions/Kaggle/BraST2020/data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', 'nii.gz', transform=val_transform)
    val_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    # model = DeepSEED(
    #     4,
    #     [8, 16, 32, 64],
    #     out_dim=3,
    #     dropout_p=0.1
    # )
    # model = SegResNet(
    #     blocks_down=[1, 2, 2, 4],
    #     blocks_up=[1, 1, 1],
    #     init_filters=16,
    #     in_channels=4,
    #     out_channels=3,
    #     dropout_prob=0.2,
    # )

    # model = VNet(
    #     spatial_dims=3,
    #     in_channels=4,
    #     out_channels=3,
    #     dropout_prob=0.1
    # )

    # model = nets.BasicUNetPlusPlus(
    #     spatial_dims=3,
    #     in_channels=4,
    #     out_channels=3,
    #     # deep_supervision=True,
    #     features=[8, 16, 32, 64, 128, 8],
    #     dropout=0.1,
    # )

    model = nets.BasicUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        # act=("MEMSWISH",{}),
        # dropout=0.05
    )

    # model = nets.UNet(
    #     spatial_dims=3,
    #     in_channels=4,
    #     out_channels=3,
    #     channels=(32, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=3
    # )

    # model.__class__.__name__ = "BasicUNet-SWISH"
    print(model)
    # exit()
    
    pl_runner = LightningRunner(model, args)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # lr_finder = LearningRateFinder(num_training_steps=8000)
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_dice',
        filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{avg_dice:.4f}',
        mode='max'
    )

    # logger = TensorBoardLogger(
    #     save_dir='.',
    #     version='dummy'
    #     # version=f'{model.__class__.__name__}'
    #     )

    trainer = Trainer(
        max_epochs=args.epochs,
        devices=[2,3],
        accelerator='gpu',
        precision=16,
        strategy='ddp',
        callbacks=[lr_monitor, checkpoint_callback],
        check_val_every_n_epoch=10,
        # logger=logger
        )

    trainer.fit(
        model= pl_runner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
