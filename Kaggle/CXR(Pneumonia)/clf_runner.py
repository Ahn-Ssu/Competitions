

def run():
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'
    os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
    os.environ["TORCH_USE_CUDA_DSA"]= 'True'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
    from pytorch_lightning.loggers import TensorBoardLogger

    from CXR_dataloader import KFold_pl_DataModule
    from clf_lightning import Classification_network
    from model import simple
    from utils.transform_generator import MONAI_transformerd

    args = EasyDict()

    # training cfg
    args.img_size = 512
    args.batch_size = 16 # mk4 32bit-4Bz = 44201
    args.epoch = 300
    args.init_lr = 1e-4
    args.lr_dec_rate = 0.01 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.05

    # model cfg

    args.seed = 411
    args.server = 'mk2'
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

    train_transform = transformer.generate_test_transform(args.intensity_cfg)
    test_transform = transformer.generate_test_transform(args.intensity_cfg)
    # train_transform = transformer.generate_train_transform(args.intensity_cfg,
    #                                                        augmentation_cfg=args.augmentation_cfg,
    #                                                        is_randAug=args.is_randAug)

    # set_track_meta(False)

    num_split = 5
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):

        pl_dataloder = KFold_pl_DataModule(
                            data_root_dir='/root/Competitions/Kaggle/CXR(Pneumonia)/data/chest_xray',
                            k_idx=fold_idx,
                            num_split=num_split,
                            split_seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=False,
                            persistent_workers=True,
                            train_transform=train_transform,
                            val_transform=test_transform
                        )

        model = simple.simple_CNN()
        
        
        pl_runner = Classification_network(network=model, args=args)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_auroc',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{val_auroc:.3f}-{val_f1score:.3f}',
                                    mode='max',
                                    # save_top_k=1,
                                    save_last=True
                                )
        # SWA = StochasticWeightAveraging(swa_lrs=5e-6,
        #                                 swa_epoch_start=0.15,
        #                                 annealing_epochs=25)
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            # version='LEARNING CHECK',
                            version=f'0.ealry_stage/{_day}/mk2 firstRun',
                            default_hp_metric=False
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=4,
                    accelerator='gpu',
                    precision='16-mixed',
                    # strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback],
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
                # ckpt_path='/root/Competitions/MICCAI/AutoPET2023/lightning_logs/5.SAW/2023-09-17/mk2)Lv2 aug + 160x2 from mk2 lv2(annealing=25)/checkpoints/last.ckpt'
            )
        
        break;

    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()