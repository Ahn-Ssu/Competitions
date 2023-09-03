
from model import late_fusion, middle_fusion


def run():
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= '2,3'
    import random
    from datetime import datetime, timezone, timedelta

    from easydict import EasyDict

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.profilers import PyTorchProfiler

    from monai.transforms import (
                            Compose,
                            OneOf,

                            LoadImaged,
                            EnsureTyped,
                            ScaleIntensityRanged,
                            Orientationd,
                            Spacingd,
                            CropForegroundd, 
                            RandSpatialCropd,
                            
                            RandFlipd,
                            RandCoarseShuffled,
                            RandGaussianNoised,
                            RandCoarseDropoutd,
                            
                        )

    from SSL_dataloader import KFold_pl_DataModule
    from model import unet_baseline
    from modelgenesis_pl import Modelgenesis_network

    args = EasyDict()

    args.img_size = (128, 128, 96)
    args.batch_size = 8
    args.epoch = 1000
    args.init_lr = 1e-3
    args.lr_dec_rate = 0.001 
    args.weight_decay = 0.05

    # preprocesing cfg
    args.CT_min = -1000
    args.CT_max = 1000
    args.CT_clip = False
    args.PET_min = 0
    args.PET_max = 40
    args.PET_max2 = 20
    args.PET_clip = False

    # model cfg
    args.hidden_dims = [32,32,64,128,256]
    args.dropout_p = 0.
    args.use_MS = False

    args.genesis_args = EasyDict()
    args.genesis_args.nonlinear_rate = 0.9  # prob of non-linear transformation
    args.genesis_args.local_rate = 0.5      # prob of local pixel shuffling
    args.genesis_args.paint_rate = 0.9      # prob of (in/out) painting 
    args.genesis_args.outpaint_rate = 0.8   # prob of outer painting
    args.genesis_args.inpaint_rate = 0.2    # prob of inner painting

    ## added to 7
    args.genesis_args.noise_rate = 0.9      # ADDED - prob of noising 
    args.genesis_args.modality = "PET" # "PET"

    # for model genesis, basic augmentation
    args.genesis_args.rotation_rate = 0.0
    args.genesis_args.flip_rate = 0.4
    args.genesis_args.norm_type = "minmax"

    args.seed = 41
    args.server = 'mk4'
    seed.seed_everything(args.seed)



    transform = Compose([
        LoadImaged(keys='img', ensure_channel_first=True),
        EnsureTyped(keys='img', track_meta=True), # when we use spaingd, track_meta shuold be true
        Spacingd(keys='img',
                 pixdim=(2.03642,  2.03642, 3.), mode=("bilinear")),
        Orientationd(keys='img', axcodes='RAS'),
        OneOf([
                ScaleIntensityRanged(keys='img',
                                    a_min=args.PET_min, a_max=args.PET_max,
                                    b_min=0, b_max=1, clip=args.PET_clip),
                ScaleIntensityRanged(keys='img',
                                    a_min=args.PET_min, a_max=args.PET_max2,
                                    b_min=0, b_max=1, clip=args.PET_clip),    
            ]),
        CropForegroundd(keys='img', source_key='img'), # source_key 'ct' or 'pet'
        RandSpatialCropd(keys='img', roi_size=args.img_size, random_size=False),
        # flip, pixel shuffling, in-out painting 
        RandFlipd(keys='img', prob=args.genesis_args.flip_rate, spatial_axis=0), 
        RandFlipd(keys='img', prob=args.genesis_args.flip_rate, spatial_axis=1),
        RandFlipd(keys='img', prob=args.genesis_args.flip_rate, spatial_axis=2),
        ]) if args.genesis_args.modality == 'PET' else Compose([
        LoadImaged(keys='img', ensure_channel_first=True),
        EnsureTyped(keys='img', track_meta=True), # when we use spaingd, track_meta shuold be true
        Spacingd(keys='img',
                 pixdim=(2.03642,  2.03642, 3.), mode=("bilinear")),
        Orientationd(keys='img', axcodes='RAS'),
        ScaleIntensityRanged(keys='img',
                                 a_min=args.CT_min, a_max=args.CT_max,
                                 b_min=0, b_max=1, clip=args.CT_clip),
        CropForegroundd(keys='img', source_key='img'),
        RandSpatialCropd(keys='img', roi_size=args.img_size, random_size=False),
    ]    )
    
    PET_additional_transform = None if args.genesis_args.modality == 'CT' else Compose([
        RandCoarseShuffled(keys='img',
                           prob=args.genesis_args.local_rate,
                        holes=100,
                        max_holes=10000,
                        spatial_size=[1,1,1], # minimum size
                        max_spatial_size=[args.img_size[0]//10, args.img_size[1]//10, args.img_size[2]//10]),
        
        OneOf([RandGaussianNoised(keys='img', prob=args.genesis_args.noise_rate, 
                                  mean=random.uniform(0.0, 0.1) if random.random() > 0.5 else 0 , 
                                  std=random.uniform(0.001, 0.1)) for i in range(100)]),
        
        OneOf([RandCoarseDropoutd(keys='img', prob=args.genesis_args.paint_rate,
                          holes=3,
                          max_holes=5,
                          fill_value=(0, 1),
                          spatial_size=[args.img_size[0]//6, args.img_size[1]//6, args.img_size[2]//6],
                          max_spatial_size=[args.img_size[0]//3, args.img_size[1]//3, args.img_size[2]//3],
                          dropout_holes=True # inner cutout
                          ),
               RandCoarseDropoutd(keys='img',prob=args.genesis_args.paint_rate,
                          holes=3,
                          max_holes=5,
                          fill_value=(0, 1),
                          spatial_size=[3*args.img_size[0]//7, 3*args.img_size[1]//7, 3*args.img_size[2]//7],
                          max_spatial_size=[4*args.img_size[0]//7, 4*args.img_size[1]//7, 4*args.img_size[2]//7],
                          dropout_holes=False # outer cutout
                          )
               ], weights=(0.8, 0.2))
    ]
    )

    num_split = 10 # training : validation = 9 : 1 
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):

        pl_dataloder = KFold_pl_DataModule(
                            modality=args.genesis_args.modality,
                            k_idx=fold_idx,
                            num_split=num_split,
                            split_seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=10,
                            pin_memory=False,
                            persistent_workers=True,
                            train_transform=transform,
                            additional=PET_additional_transform,
                        )

        model = unet_baseline.UNet(
                            input_dim=1,
                            out_dim=1,
                            hidden_dims=args.hidden_dims, # 16 32 32 64 128 is default setting of Monai
                            spatial_dim=3,
                            dropout_p=args.dropout_p
                        )

        
        pl_runner = Modelgenesis_network(network=model, args=args)#.load_from_checkpoint('/root/Competitions/MICCAI/AutoPET2023/lightning_logs/IntensityRange/2023-06-28/CT=(-100, 400), PET=(0, 40) || UNet_lateF(16,256) w He - GPU devices[2,3]/checkpoints/UNet_lateF-epoch=182-train_loss=0.3444-val_dice=0.6879.ckpt', network=model, args=args)
        

        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_loss',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{val_loss:.4f}',
                                    mode='min',
                                    # save_top_k=1,
                                    save_last=True
                                )
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            default_hp_metric=False,
                            # version='LEARNING CHECK',
                            version=f'10.ModelGenesis/{_day}/PET-AutoPET+Extra data'
                        )
        profiler = PyTorchProfiler()

        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[0],
                    accelerator='gpu',
                    # precision='16-mixed',
                    # strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback],
                    check_val_every_n_epoch=5,
                    # log_every_n_steps=1,
                    logger=logger,
                    # auto_lr_find=True
                    # accumulate_grad_batches=2
                    # profiler='advanced', #advanced
                    # profiler=profiler
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