def run():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= '1'
    from easydict import EasyDict
    from datetime import datetime, timezone, timedelta

    from lightning_fabric.utilities import seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    
    from dataloader import KFold_pl_DataModule
    from model import molp
    from reg_pl import Regression_network
    
    # training cfg
    args = EasyDict()
    args.batch_size = 128
    args.epoch = 1000
    args.init_lr = 1e-3
    args.lr_dec_rate = 0.01 # 기존에 쓰던건 0.001로 많이 내려가게 했었음 
    args.weight_decay = 0.05
    
    args.MLM = True
    args.train_label = 'MLM' if args.MLM else 'HLM'

    # model cfg
    args.num_atom_f = 34
    args.num_mol_f = 40
    args.gnn_hidden_dims = [64, 128, 256, 512]
    args.d_model = args.num_mol_f + sum(args.gnn_hidden_dims)
    args.FF_hidden_dim = 2048
    args.dropout_p = 0.09

    args.seed = 41
    args.server = 'mk3'
    seed.seed_everything(args.seed)
    
    # No transformers yet 
    
    num_split = 5
    KST = timezone(timedelta(hours=9))
    start = datetime.now(KST)
    _day = str(start)[:10]

    for fold_idx in range(num_split):
        
        pl_dataloder = KFold_pl_DataModule(
                            train_df='/root/Competitions/DACON/4. JUMP AI 2023/data/new_train.csv',
                            k_idx=fold_idx,
                            num_split=num_split,
                            split_seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=8,
                            pin_memory=False,
                            persistent_workers=True,
                            train_transform=None,
                            val_transform=None
                        )
        
        model = molp.Molp(num_atom_f=args.num_atom_f,
                          gnn_hidden_dims=args.gnn_hidden_dims,
                          num_mol_f=args.num_mol_f,
                          d_model=args.d_model,
                          FF_hidden_dim=args.FF_hidden_dim,
                          output_dim=1,
                          dropout_p=args.dropout_p )
        
        args.z_model_arch = str(model)
        
        pl_runner = Regression_network(network=model, args=args)#.load_from_checkpoint('/root/Competitions/MICCAI/AutoPET2023/lightning_logs/IntensityRange/2023-06-28/CT=(-100, 400), PET=(0, 40) || UNet_lateF(16,256) w He - GPU devices[2,3]/checkpoints/UNet_lateF-epoch=182-train_loss=0.3444-val_dice=0.6879.ckpt', network=model, args=args)
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
                            version='LEARNING CHECK',
                            # version=f'0.1st_RUN/{_day}/{args.train_label}/GIN+Transformer from scratch',
                            default_hp_metric=False
                        )
        
        trainer = Trainer(
                    max_epochs=args.epoch,
                    devices=[0],
                    accelerator='gpu',
                    # precision='16-mixed',
                    # strategy=DDPStrategy(find_unused_parameters=True), # late fusion ㅎㅏㄹㄸㅐ ㅋㅕㄹㅏ..
                    callbacks=[lr_monitor, checkpoint_callback],
                    # check_val_every_n_epoch=2,
                    check_val_every_n_epoch=5,
                    # log_every_n_steps=1,
                    logger=logger,
                    # auto_lr_find=True
                    # accumulate_grad_batches=2
                    # profiler='simple', #advanced
                )
        

        
        trainer.fit(
                model= pl_runner,
                datamodule= pl_dataloder,
            )
        
        break;
    # fold iteration END
    print(f'execution done --- time cost: [{datetime.now(KST) - start}]')


if __name__ == '__main__':
    run()