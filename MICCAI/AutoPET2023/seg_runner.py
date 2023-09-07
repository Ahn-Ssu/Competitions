

def run():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
    from pytorch_lightning.loggers import TensorBoardLogger

    from dataloader import KFold_pl_DataModule
    from model import unet_baseline, late_fusion, middle_fusion
    from seg_pl import Segmentation_network
    from utils.transform_generator import MONAI_transformerd

    from monai.networks.nets import basic_unet

    args = EasyDict()

    # training cfg
    args.img_size = 192
    args.batch_size = 2 # mk4 32bit-4Bz = 44201
    args.epoch = 1000
    args.init_lr = 1e-4
    args.lr_dec_rate = 0.001 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.05

    # model cfg
    args.hidden_dims = [32,32,64,128,256]
    args.dropout_p = 0.
    args.use_MS = False

    args.seed = 41
    args.server = 'mk4'
    seed.seed_everything(args.seed)


    all_key = ['ct','pet','label']
    input_key = ['ct','pet']
    args.aug_Lv = 2
    transformer = MONAI_transformerd(aug_Lv=args.aug_Lv,
                                     all_key=all_key, input_key=input_key, 
                                     input_size=(args.img_size, args.img_size, args.img_size))
    intensity_cfg, augmentation_cfg = transformer.get_CFGs()
    args.intensity_cfg = intensity_cfg
    args.augmentation_cfg = augmentation_cfg
    args.is_randAug = False

    test_transform = transformer.generate_test_transform(args.intensity_cfg)
    train_transform = transformer.generate_train_transform(args.intensity_cfg,
                                                           augmentation_cfg=args.augmentation_cfg,
                                                           is_randAug=args.is_randAug)

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
                            spatial_dim=3,
                            input_dim=2,
                            out_dim=2,
                            hidden_dims=args.hidden_dims, # 16 32 32 64 128 is default setting of Monai
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
        args.z_model_arch = str(model)
        
        pl_runner = Segmentation_network(network=model, args=args)#.load_from_checkpoint('/root/Competitions/MICCAI/AutoPET2023/lightning_logs/IntensityRange/2023-06-28/CT=(-100, 400), PET=(0, 40) || UNet_lateF(16,256) w He - GPU devices[2,3]/checkpoints/UNet_lateF-epoch=182-train_loss=0.3444-val_dice=0.6879.ckpt', network=model, args=args)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_dice',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{val_dice:.4f}',
                                    mode='max',
                                    # save_top_k=1,
                                    save_last=True
                                )
        SWA = StochasticWeightAveraging(swa_lrs=args.init_lr/2,
                                        swa_epoch_start=0.6,
                                        annealing_epochs=args.epoch//20)
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            # version='LEARNING CHECK',
                            version=f'5.SAW/{_day}/mk4)Lv2 aug + 192 + corrected middle fusion',
                            default_hp_metric=False
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[0,1],
                    accelerator='gpu',
                    precision='16-mixed',
                    # strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback, SWA],
                    # check_val_every_n_epoch=2,
                    check_val_every_n_epoch=10,
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