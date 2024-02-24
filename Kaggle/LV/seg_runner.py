def run():
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= '2,3'
    # os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
    # os.environ["TORCH_USE_CUDA_DSA"]= '1'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
    from pytorch_lightning.loggers import TensorBoardLogger

    from MRI_dataloader import KFold_pl_DataModule
    from Segmentation_lightning import Segmentation_network
    from utils.transform_generator import MONAI_transformerd
    
    import torch
    import torch._dynamo
    # from monai.networks.nets import basic_unet
    from models.unet import MyUNet

    torch._dynamo.config.suppress_errors = True
    args = EasyDict()

# 134.38036809815952 110.02453987730061 157.11656441717793
    # training cfg
    args.img_size = (144, 144, 144)
    args.batch_size = 1 # mk4 32bit-4Bz = 44201
    args.epoch = 1000
    args.init_lr = 1e-3
    args.lr_dec_rate = 0.001 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.005

    # model cfg
    args.hidden_dims = [32,32,64,128,256, 32]
    args.dropout_p = 0.

    args.seed = 41
    args.server = 'mk4'
    seed.seed_everything(args.seed)


    all_key = ['image','label']
    input_key = 'image'
    args.aug_Lv = 1
    
    T = MONAI_transformerd(aug_Lv=args.aug_Lv,
                                     all_key=all_key, input_key=input_key, 
                                     input_size=args.img_size)
    basic_cfg, augmentation_cfg = T.get_CFGs()
    args.basic_cfg = basic_cfg
    args.augmentation_cfg = augmentation_cfg
    args.is_randAug = False

    train_transform = T.generate_train_transform(basic_cfg=basic_cfg, 
                                                 augmentation_cfg=augmentation_cfg,
                                                 is_randAug=args.is_randAug)
    test_transform = T.generate_test_transform(basic_cfg=basic_cfg)

    # set_track_meta(False)

    num_split = 5
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):

        pl_dataloder = KFold_pl_DataModule(
                            k_idx=fold_idx,
                            num_split=num_split,
                            split_seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=16,
                            pin_memory=False,
                            persistent_workers=True, # persistent_workers option needs num_workers > 0
                            train_transform=train_transform,
                            val_transform=test_transform
                        )

        model = MyUNet(3, 1, 20, (32, 32, 64, 128, 256), (2, 2, 2, 2), num_res_units=2, adn_ordering="NA")
        
        # print(model)
        args.z_model_arch = str(model)
        
        pl_runner = Segmentation_network(network=model, args=args)#.load_from_checkpoint('/root/Competitions/MICCAI/AutoPET2023/lightning_logs/IntensityRange/2023-06-28/CT=(-100, 400), PET=(0, 40) || UNet_lateF(16,256) w He - GPU devices[2,3]/checkpoints/UNet_lateF-epoch=182-train_loss=0.3444-val_dice=0.6879.ckpt', network=model, args=args)
        
      
        # pl_runner = torch.compile(pl_runner)
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_dice',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{val_dice:.4f}',
                                    mode='max',
                                    # save_top_k=1,
                                    save_last=True
                                )
        # SWA = StochasticWeightAveraging(swa_lrs=args.init_lr/2,
        #                                 swa_epoch_start=0.6,
        #                                 annealing_epochs=args.epoch//20)
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            # version='SegFault-2',
                            # version='LEARNING CHECK',
                            version=f'firstRun/{_day}/mk5)mgz-run',
                            default_hp_metric=False
                        )
        
        trainer = Trainer( # https://github.com/Lightning-AI/lightning/issues/12398
                    max_epochs=args.epoch,
                    devices=4,
                    accelerator='gpu',
                    precision='16-mixed',
                    strategy=DDPStrategy(find_unused_parameters=False), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    benchmark=False,
                    callbacks=[lr_monitor, checkpoint_callback],
                    # check_val_every_n_epoch=2,
                    check_val_every_n_epoch=5,
                    # log_every_n_steps=1,
                    logger=logger,
                    # auto_lr_find=True
                    # accumulate_grad_batches=2
                    # profiler='simple', #advanced
                    # num_sanity_val_steps=0
                )
        

        
        trainer.fit(
                model= pl_runner,
                datamodule= pl_dataloder,
                # If we wanna keep training use this code line 
                # ckpt_path='/root/snsb/lightning_logs/1.firstRun/2023-11-15/mk5)High_LR-seg_crop+adabelif/checkpoints/last.ckpt'
            )
        
        break;

    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()