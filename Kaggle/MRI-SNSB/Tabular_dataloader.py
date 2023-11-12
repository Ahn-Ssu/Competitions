import glob
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, AsDiscrete, Resize, EnsureTyped, ScaleIntensityRanged, NormalizeIntensityd


categorical = ['Sex(M=1, F=2)', 'Hypertension', 'Diabete', 'Hyperlipidemia']
numerical = ['Age', 'Education', 'CDR', 'GDS', 'MMSE_Reg',
       'MMSE_Time', 'MMSE_Place', 'MMSE_Recall', 'MMSE_Attention/Calc',
       'MMSE_Lanugage', 'MMSE_Drawing', 'MMSE_Total',
       'SNSB_Attention', 'SNSB_Language', 'SNSB_Visuospatial', 'SNSB_Memory', 'SNSB_Frontal']

class Tabular_dataloader(Dataset):
    def __init__(self, df:pd.DataFrame) -> None:
        super().__init__()
        
        self.categorical = df[categorical].values
        self.numerical = df[numerical].values

    def __getitem__(self, index):
        return self.get_case(self.categorical[index], self.numerical[index])

    def __len__(self):
        return len(self.categorical)
    
    def get_case(self, cat_f, num_f):
        
        cat_f = torch.tensor(cat_f, dtype=torch.long)
        num_f = torch.tensor(num_f, dtype=torch.float)
        
        data_d = {
            'cat_f': cat_f,
            'num_f': num_f
        }
            
        return data_d
    


class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 k_idx: int =1, # fold index
                 num_split: int = 5, # fold number, if k=1 then return the whole data
                 split_seed: int = 41,
                 batch_size: int = 2, 
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 persistent_workers: bool=True,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_data = None
        self.val_data = None
        # self.test_data = None

        self.setup()

    def setup(self, stage=None) -> None:
        if not self.train_data and not self.val_data:
            # tarin data
            
            df = pd.read_csv('/root/Competitions/Kaggle/MRI-SNSB/data/pre-processed_no-smoking&alcohol.csv')
            
            kf = StratifiedKFold(n_splits=10,
            shuffle=True,
            random_state=411)

            all_splits = [k for k in kf.split(df, df['Sex(M=1, F=2)'])]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train = df.iloc[train_idx]
            val = df.iloc[val_idx]
            
            self.train_data = Tabular_dataloader(train)
            self.val_data   = Tabular_dataloader(val)
            

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory,
                          drop_last=True)
                          
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size, # self.hparams.batch_size, # when PT -> bz else 1 
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
    
    # def test_dataloader(self):
    #     return DataLoader(self.test_data,
    #                       batch_size=self.hparams.batch_size, # self.hparams.batch_size, # when PT -> bz else 1 
    #                       shuffle=False,
    #                       num_workers=self.hparams.num_workers,
    #                       persistent_workers=self.hparams.persistent_workers,
    #                       pin_memory=self.hparams.pin_memory)
    


if __name__ == "__main__":

    pl_dataloader = KFold_pl_DataModule(batch_size=1,
                                        num_workers=1)

    val_dataloader = pl_dataloader.train_dataloader()

    for idx, batch in enumerate(val_dataloader):
        cat_f, num_f = batch['cat_f'], batch['num_f']
        print(cat_f.size(), num_f.size())