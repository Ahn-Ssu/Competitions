

def run():
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'
#    os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
#    os.environ["TORCH_USE_CUDA_DSA"]= 'True'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    from Tabular_dataloader import KFold_pl_DataModule
    from Generation_lightning import Generation_networks
    from model import VAE

    args = EasyDict()

    # training cfg
    args.batch_size = 4 # mk4 32bit-4Bz = 44201
    args.epoch = 200
    args.init_lr = 1e-4
    args.lr_dec_rate = 0.01 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.005

    # model cfg

    args.seed = 411
    args.server = 'Mk3'
    seed.seed_everything(args.seed)


    # set_track_meta(False)

    num_split = 10
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):

        pl_dataloder = KFold_pl_DataModule(
                            batch_size=args.batch_size,
                            num_workers=8,
                            k_idx=fold_idx,
                            pin_memory=False,
                            persistent_workers=True,
                        )

        model = VAE.MLP_VAE(data_dim=21)
        
        
        pl_runner = Generation_networks(network=model, args=args)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        checkpoint_callback = ModelCheckpoint(
                                    monitor='val_loss',
                                    filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{val_loss:.3f}-{train_loss:.3f}',
                                    mode='min',
                                    # save_top_k=1,
                                    # save_last=True
                                )
        
        logger = TensorBoardLogger(
                            save_dir='.',
                            # version='LEARNING CHECK',
                            version=f'1.FirstRun/{_day}/w dropout',
                            default_hp_metric=False
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=1,
                    accelerator='gpu',
                    # precision=16,
                    # strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback],
                    check_val_every_n_epoch=5,
                    logger=logger,
                )

        
        trainer.fit(
                model= pl_runner,
                datamodule= pl_dataloder,
                # ckpt_path='/root/Competitions/MICCAI/AutoPET2023/lightning_logs/5.SAW/2023-09-17/mk2)Lv2 aug + 160x2 from mk2 lv2(annealing=25)/checkpoints/last.ckpt'
            )
        
        break;

    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()
