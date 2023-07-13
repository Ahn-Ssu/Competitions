
from model import late_fusion, middle_fusion


def run():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= '2,3'
    from datetime import datetime, timezone, timedelta

    from easydict import EasyDict

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.profilers import PyTorchProfiler

    from monai.transforms import (
                            Compose,

                            LoadImaged,
                            EnsureTyped,
                            ScaleIntensityRanged,
                            Orientationd,
                            CropForegroundd, 
                            RandSpatialCropd,
                        )

    from dataloader import KFold_pl_DataModule
    from model import unet_baseline
    from modelgenesis_pl import Modelgenesis_network

    args = EasyDict()

    args.img_size = 128
    args.batch_size = 4
    args.epoch = 1000
    args.init_lr = 1e-2
    args.lr_dec_rate = 0.0001 
    args.weight_decay = 0.05

    # preprocesing cfg
    args.CT_min = -600
    args.CT_max = 400
    args.PET_min = 0
    args.PET_max = 40 

    # model cfg
    args.hidden_dims = [16,32,64,128,256]
    args.dropout_p = 0.
    args.use_MS = False

    args.genesis_args = EasyDict()
    args.genesis_args.nonlinear_rate = 0.9  # prob of non-linear transformation
    args.genesis_args.local_rate = 0.5      # prob of local pixel shuffling
    args.genesis_args.paint_rate = 0.9      # prob of (in/out) painting 
    args.genesis_args.outpaint_rate = 0.8   # prob of outer painting
    args.genesis_args.inpaint_rate = 0.2    # prob of inner painting

    ## added to 
    args.genesis_args.noise_rate = 0.9      # ADDED - prob of noising 
    args.genesis_args.modality = "PET" # "PET"

    # for model genesis, basic augmentation
    args.genesis_args.rotation_rate = 0.0
    args.genesis_args.flip_rate = 0.4
    args.genesis_args.norm_type = "minmax"

    args.seed = 41
    seed.seed_everything(args.seed)


    all_key = ['ct','pet','label']

    transform = Compose([
        LoadImaged(keys=all_key, ensure_channel_first=True),
        EnsureTyped(keys=all_key, track_meta=False),
        Orientationd(keys=all_key, axcodes='RAS'),
        ScaleIntensityRanged(keys='ct',
                                 a_min=args.CT_min, a_max=args.CT_max,
                                 b_min=0, b_max=1, clip=True),
        ScaleIntensityRanged(keys='pet',
                                a_min=args.PET_min, a_max=args.PET_max,
                                b_min=0, b_max=1, clip=True),

        CropForegroundd(keys=all_key, source_key='pet'), # source_key 'ct' or 'pet'
        RandSpatialCropd(keys=all_key, roi_size=[args.img_size,args.img_size,args.img_size], random_size=False)
    ]
    )

    num_split = 10 # training : validation = 9 : 1 
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
                            train_transform=transform,
                            val_transform=transform
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
                            version=f'Modelgenesis/{_day}/PET=(0, 40) || UNet(32,512) w He - GPU devices[2,3]'
                        )
        profiler = PyTorchProfiler()

        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[0,1],
                    accelerator='gpu',
                    precision='16-mixed',
                    # strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback],
                    check_val_every_n_epoch=3,
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