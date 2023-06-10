
if __name__ == '__main__':

    from easydict import EasyDict

    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
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


    from lightning import LightningRunner
    from data_loader import *
    from model.models import *
    from torch.utils.data import DataLoader

    args = EasyDict()

    args.img_size = 368

    args.batch_size = 32
    args.epochs = 80
    args.init_lr = 8e-5
    args.weight_decay = 0.05

    args.seed = 1120
    

    seed_everything(args.seed)

    

    all_img_list = glob.glob('./aug_data/train/*/*')
    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])
 

    train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=args.seed)

    le = preprocessing.LabelEncoder()
    train['label'] = le.fit_transform(train['label'])
    val['label'] = le.transform(val['label'])

    args.prior = df.label.value_counts().to_frame().sort_index().values
    args.num_cls = len(le.classes_)


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
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


    train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform_4_origin)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)

    val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

    model = BaseModel(len(le.classes_))
    pl_runner = LightningRunner(model, args)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    
    # lr_finder = LearningRateFinder(num_training_steps=8000)
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_f1',
        filename=f'{model.__class__.__name__}'+'-{epoch:03d}-{train_loss:.4f}-{avg_f1:.4f}',
        mode='max'
    )

    logger = TensorBoardLogger(
        save_dir='.',
        version='LEARNING CHECK',
        # version=f'[S.7] --m ConvNeXt, --d A, --t GV --L F long || lr=[{args.init_lr}], img=[{args.img_size}], bz=[{args.batch_size}]'
        )

    trainer = Trainer(
        max_epochs=args.epochs,
        devices=[0],
        accelerator='gpu',
        precision='16-mixed',
        # strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[lr_monitor, checkpoint_callback],
        # check_val_every_n_epoch=2,py
        check_val_every_n_epoch=2,
        # log_every_n_steps=1,
        logger=logger,
        # auto_lr_find=True
        # accumulate_grad_batches=2
        )


    trainer.fit(
        model= pl_runner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        # datamodule=pl_data
    )
