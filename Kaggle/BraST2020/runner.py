
if __name__ == '__main__':

    from easydict import EasyDict
    from torch.utils.data import DataLoader
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    

    from monai.transforms import (
                        Activations,
                        Activationsd,
                        AsDiscrete,
                        AsDiscreted,
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

    from monai.networks.nets import SegResNet
    from data_loader import MRI_dataset, ConvertToMultiChannelBasedOnBratsClassesd
    from lighting import LightningRunner


    args = EasyDict()

    args.batch_size = 2
    args.epochs = 40
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
    train_data = MRI_dataset('./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', 'nii', train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    val_data = MRI_dataset('./data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', 'nii', transform=val_transform)
    val_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)

    pl_runner = LightningRunner(model, args)

    trainer = Trainer(
        max_epochs=100,
        devices=[1,2],
        accelerator='gpu',
        precision=16,
        strategy=DDPStrategy(find_unused_parameters=False)
        )

    trainer.fit(
        model= pl_runner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )



    