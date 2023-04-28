import os
from sklearn import preprocessing


# def replace_module(modules:nn.Module, target, source):
#         for name, child in modules.named_children():
#             if isinstance(child, target):
#                 modules._modules[name] = source()
#             # elif isinstance(child, nn.Sequential):
#             else: 
#                 replace_module(child, target, source)




if __name__ == '__main__':

    from easydict import EasyDict
    from lightning_fabric.utilities.seed import seed_everything
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
    from pytorch_lightning import tuner as Tuner

    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    import glob    
    import pandas as pd


    from lighting import LightningRunner
    from data_loader import *
    # from kfold_pl_data import KFold_pl_DataModule
    from model.models import *
    from torch.utils.data import DataLoader
    from stratifiedKfold_pl_data import KFold_pl_DataModule



    args = EasyDict()

    args.img_size = 368

    args.batch_size = 32
    args.epochs = 80
    args.init_lr = 8e-5
    args.weight_decay = 0.05

    args.seed = 41
    

    seed_everything(args.seed)
    


    train_transform_4_origin = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.AdvancedBlur(),
                            A.ColorJitter(),
                            A.GaussNoise(),
                            A.OpticalDistortion(distort_limit=(-0.3, 0.3), shift_limit=0.5, p=0.5),
                            A.HorizontalFlip(),
                            A.Affine(scale=(0.9, 2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), shear=(-20,20)),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            A.ElasticTransform(alpha=300),
                            ToTensorV2()
                            ])
    
    test_transform = A.Compose([
                            A.Resize(args.img_size, args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
    
    num_split = 5 

    for k_idx in range(num_split):
        pl_dataFolder = KFold_pl_DataModule(data_dir='./aug_data/train/*/*',
                            k_idx=k_idx,
                            num_split=num_split,
                            split_seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=False,
                            persistent_workers=True,
                            train_transform=train_transform_4_origin,
                            val_transform=test_transform)

        pl_dataFolder.setup(stage=None)
        # exit()
        model = BaseModel(pl_dataFolder.num_cls)
        pl_runner = LightningRunner(model, args)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        checkpoint_callback = ModelCheckpoint(
            monitor='avg_f1',
            filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{avg_f1:.4f}',
            mode='max'
        )

        logger = TensorBoardLogger(
            save_dir='.',
            version='LEARNING CHECK',
            # version=f'[{k_idx+1} Fold] --backbone ConvNeXt_t, --data Aug, --transform Geo+Value || lr=[{args.init_lr}], img_size = [{args.img_size}], bz=[{args.batch_size}]'
            )

        trainer = Trainer(
            max_epochs=args.epochs,
            devices=[0],
            accelerator='gpu',
            precision='16-mixed',
            # strategy=DDPStrategy(find_unused_parameters=False),
            callbacks=[lr_monitor, checkpoint_callback],
            # check_val_every_n_epoch=2,
            check_val_every_n_epoch=2,
            # log_every_n_steps=1,
            logger=logger,
            # auto_lr_find=True
            # accumulate_grad_batches=2
            )

        trainer.fit(
            model= pl_runner,
            # train_dataloaders=train_loader,
            # val_dataloaders=val_loader
            datamodule=pl_dataFolder
        )

    # fold iteration END
