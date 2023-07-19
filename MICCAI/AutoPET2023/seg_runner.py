

def run():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= '2,3' # '0,1'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    from monai.transforms import (
                            Compose,
                            OneOf,

                            LoadImaged,
                            EnsureTyped,
                            ScaleIntensityRanged,
                            ScaleIntensityRangePercentilesd,
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

    from dataloader import KFold_pl_DataModule
    from model import unet_baseline, late_fusion, middle_fusion
    from seg_pl import Segmentation_network

    from monai.networks.nets import basic_unet

    args = EasyDict()

    # training cfg
    args.img_size = 128
    args.batch_size = 1
    args.epoch = 200
    args.init_lr = 1e-4
    args.lr_dec_rate = 0.001 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.05

    # preprocesing cfg
    args.CT_min = -600
    args.CT_max = 400
    args.PET_min = 0
    args.PET_max = 40
    args.PET_lower = 0.
    args.PET_upper = 98

    # model cfg
    args.hidden_dims = [32,32,64,128,256]
    args.dropout_p = 0.
    args.use_MS = False

    args.seed = 41
    args.server = 'mk3'
    seed.seed_everything(args.seed)


    all_key = ['ct','pet','label']

    test_transform = Compose([
        LoadImaged(keys=all_key, ensure_channel_first=True),
        EnsureTyped(keys=all_key, track_meta=False),
        Orientationd(keys=all_key, axcodes='RAS'),
        ScaleIntensityRanged(keys='ct',
                                 a_min=args.CT_min, a_max=args.CT_max,
                                 b_min=0, b_max=1, clip=True),
        # ScaleIntensityRanged(keys='pet',
        #                         a_min=args.PET_min, a_max=args.PET_max,
        #                         b_min=0, b_max=1, clip=True),
        ScaleIntensityRangePercentilesd(keys='pet',
                                        lower=args.PET_lower, upper=args.PET_upper,
                                        b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=all_key, source_key='pet'), # source_key 'ct' or 'pet'
    ]
    )

    train_transform = Compose([
            LoadImaged(keys=all_key, ensure_channel_first=True),
            EnsureTyped(keys=all_key, track_meta=False), # for training track_meta=False, monai.data.set_track_meta(false)
            Orientationd(keys=all_key, axcodes='RAS'),
            ScaleIntensityRanged(keys='ct',
                                 a_min=args.CT_min, a_max=args.CT_max,
                                 b_min=0, b_max=1, clip=True),
            # ScaleIntensityRanged(keys='pet',
            #                      a_min=args.PET_min, a_max=args.PET_max,
            #                      b_min=0, b_max=1, clip=True),
            ScaleIntensityRangePercentilesd(keys='pet',
                                        lower=args.PET_lower, upper=args.PET_upper,
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
            # )000
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

        model = middle_fusion.UNet_middleF(
                            input_dim=3,
                            out_dim=2,
                            hidden_dims=args.hidden_dims, # 16 32 32 64 128 is default setting of Monai
                            spatial_dim=3,
                            dropout_p=args.dropout_p,
                            use_MS=args.use_MS
                        )
        
        # model = late_fusion.UNet_lateF(
        #                     input_dim=3,
        #                     out_dim=2,
        #                     hidden_dims=args.hidden_dims, # 16 32 32 64 128 is default setting of Monai
        #                     spatial_dim=3,
        #                     dropout_p=args.dropout_p,
        #                     use_MS=args.use_MS
        #                 )
        
        # print(model)
        
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
                            version=f'2.Intensity/{_day}/PET)Percentiles, middle fusion',
                            default_hp_metric=False
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[0,1],
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