

def run():
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'
#    os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
#    os.environ["TORCH_USE_CUDA_DSA"]= 'True'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from pytorch_lightning.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    from CXR_dataloader import KFold_pl_DataModule
    from reg_lightning import Regression_Network
    from model import Reg_CNN
    from utils.transform_generator import MONAI_transformerd

    args = EasyDict()

    # training cfg
    args.img_size = (128, 256)
    args.batch_size = 16 # mk4 32bit-4Bz = 44201
    args.epoch = 500
    args.init_lr = 1e-4
    args.lr_dec_rate = 0.01 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.00

    # model cfg

    args.seed = 411
    args.server = 'KTL'
    seed.seed_everything(args.seed)


    all_key = ['image']
    input_key = ['image']
    args.aug_Lv = 1
    transformer = MONAI_transformerd(aug_Lv=args.aug_Lv,
                                     all_key=all_key, input_key=input_key, 
                                     input_size=args.img_size)
    intensity_cfg, augmentation_cfg = transformer.get_CFGs()
    args.intensity_cfg = intensity_cfg
    args.augmentation_cfg = augmentation_cfg
    args.is_randAug = False

    test_transform = transformer.generate_test_transform(args.intensity_cfg)
    train_transform = transformer.generate_train_transform(args.intensity_cfg, args.augmentation_cfg, False)
    
    # set_track_meta(False)
    num_split = 5
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):

        pl_dataloder = KFold_pl_DataModule(
                            batch_size=args.batch_size,
                            num_workers=16,
                            pin_memory=False,
                            persistent_workers=True,
                            train_transform=train_transform,
                            val_transform=test_transform
                        )

        model = Reg_CNN.simple_CNN()
        
        
#         pl_runner = Regression_Network.load_from_checkpoint('/home/pwrai/userarea/spineTeam/default/1.baseline/2023-10-31/ResNet34_128x256(MAE)/checkpoints/simple_CNN-epoch=297-train_loss=0.079-val_auroc=0.957-val_f1score=0.958-val_r2=0.737.ckpt', network=model, args=args)
        pl_runner = Regression_Network(network=model, args=args)
        lr_monitor = LearningRateMonitor(logging_interval='step')
#         pl_runner = pl_runner.

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_auroc',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.3f}-{val_auroc:.3f}-{val_f1score:.3f}-{val_r2:.3f}',
                                    mode='max',
                                    # save_top_k=1,
                                    save_last=True
                                )
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            # version='LEARNING CHECK',
                            version=f'1.baseline/{_day}/ResNet152_128x256(C101+C103, MAE)',
                            default_hp_metric=False
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[1],
                    accelerator='gpu',
                    precision=16,
                    callbacks=[lr_monitor, checkpoint_callback],
                    check_val_every_n_epoch=1,
                    logger=logger,
                    num_sanity_val_steps=0
                )

        
        trainer.fit(
                model= pl_runner,
                datamodule= pl_dataloder,
#                 ckpt_path='/home/pwrai/userarea/spineTeam/default/1.baseline/2023-10-31/ResNet34_128x256(MAE)/checkpoints/simple_CNN-epoch=297-train_loss=0.079-val_auroc=0.957-val_f1score=0.958-val_r2=0.737.ckpt'
            )
        
        break;

    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()
