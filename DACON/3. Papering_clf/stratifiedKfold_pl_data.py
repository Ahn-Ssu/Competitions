import os
import pytorch_lightning as pl

import glob
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing

from data_loader import CustomDataset

class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = None,
                 k_idx: int =1, # fold index
                 num_split: int = 5, # fold number, if k=1 then return the whole data
                 split_seed: int = 41,
                 batch_size: int = 2, 
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 persistent_workers: bool=True,
                 train_transform=None,
                 val_transform =None
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_data = None
        self.val_data = None
        self.num_cls = 0

    def setup(self, stage: str) -> None:
        if not self.train_data and not self.val_data:
            all_img_list = glob.glob(self.hparams.data_dir)
            df = pd.DataFrame(columns=['img_path', 'label'])
            df['img_path'] = all_img_list
            df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])

            kf = StratifiedKFold(n_splits=self.hparams.num_split,
                       shuffle=True,
                       random_state=self.hparams.split_seed)
            
            all_splits = [k for k in kf.split(df, df['label'])]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train = df.iloc[train_idx]
            val = df.iloc[val_idx]

            le = preprocessing.LabelEncoder()
            le.fit(df['label'])
            train['label'] = le.transform(train['label'])
            val['label'] = le.transform(val['label'])
            self.num_cls = len(le.classes_)


            
            self.train_data = CustomDataset(train['img_path'].values, train['label'].values, self.hparams.train_transform)
            self.val_data = CustomDataset(val['img_path'].values, val['label'].values, self.hparams.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size//2,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
