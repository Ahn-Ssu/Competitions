import glob
import pandas as pd
import pytorch_lightning as pl

import numpy as np

import torch
import torch.nn.functional as F

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, AsDiscrete, Resize, EnsureTyped, ScaleIntensityRanged, NormalizeIntensityd


class CXR_dataset(Dataset):
    def __init__(self, data_dir= None, image_path=None, cls=None, cobbs=None,  transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_path = image_path
        self.cls = cls
        self.cobbs = cobbs
        self.transform = transform
        self.onehot = AsDiscrete(to_onehot=4)
        
        self.input_key ='image'
        self.loader = Compose([
        LoadImaged(keys=self.input_key, image_only=True, ensure_channel_first=True),
        EnsureTyped(keys=self.input_key, device=None, track_meta=False)])
        
        self.scaler = Compose([
            ScaleIntensityRanged(keys=self.input_key,
                             a_max=255.0, a_min=0., b_max=1, b_min=0, clip=True),
#         HistogramNormalized(keys=self.input_key),
        NormalizeIntensityd(keys=self.input_key, subtrahend=0.4309305409308814, divisor=0.25163892359754375)
        ])

    def __getitem__(self, index):
        return self.get_SCANS(self.image_path[index], self.cls[index], self.cobbs[index])

    def __len__(self):
        return len(self.image_path)
    
    def get_SCANS(self, path:str, cls, cobbs):
        path = path.split('/')[-1].split('.')[0]
        
        if 'C101' in path:
            data_dir = f'{self.data_dir}/C101'
        elif 'C103' in path:
            data_dir = f'{self.data_dir}/C103'

        # C101 저장 에러로 임시사용
        y = torch.tensor(cls)
        class_onehot = self.onehot(y).astype(torch.long)
        
        cobbs = torch.tensor(cobbs, dtype=torch.float)
        radian = (np.pi * cobbs) / 180.
        sin = torch.sin(radian)
        cos = torch.cos(radian)
        # radian = torch.atan2(sin_theta, cos_theta)
        # cobbs = (radian *180.) / np.pi
        
        path_d = {
            'image': f'{data_dir}/{path}.png',
            'y':y,
            'class_onehot':class_onehot,
            'cobbs':cobbs,
            'sincos':torch.stack([sin, cos], dim=0)
        }

        try:
            data_d = self.transform(path_d)
#             data_d = self.loader(path_d)
        except Exception as err:
            print(err, path_d)
            exit()

        if isinstance(data_d, list):
                data_d = data_d[0]
                
        c, w, h = data_d['image'].size()
        
#         ratio = h/w
        
#         target_h = 1024
#         target_w = int(target_h/ratio)
        
#         data_d['image'] = F.interpolate(data_d['image'].unsqueeze(0), size=(target_w, target_h), mode='bilinear').squeeze(0)
        
#         data_d = self.scaler(data_d)
        
        if c == 3 :
            data_d['image'] = data_d['image'][0:1]
            
        return data_d
    


class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root_dir: str = '/home/pwrai/userarea/spineTeam/data',
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
        self.test_data = None

        self.setup()

    def setup(self, stage=None) -> None:
        if not self.train_data and not self.val_data and not self.test_data:
            # tarin data
            train_df = pd.read_csv(f'{self.hparams.data_root_dir}/train_split_c101.csv')
            train_df1 = pd.read_csv(f'{self.hparams.data_root_dir}/train_split_c103.csv')
            train_df = pd.concat([train_df, train_df1], axis=0)
            train_files = train_df['File_ID'].values
            train_cls = train_df['class'].values 
            train_cobbs = train_df['Cobbs angle'].values
            
            
            test_df = pd.read_csv(f'{self.hparams.data_root_dir}/test_split_c101.csv')
            test_df1 = pd.read_csv(f'{self.hparams.data_root_dir}/test_split_c103.csv')
            test_df = pd.concat([test_df, test_df1], axis=0)
            test_files = test_df['File_ID'].values
            test_cls = test_df['class'].values
            test_cobbs = test_df['Cobbs angle'].values

            # in AI device data, we need the following codes
            # kf = StratifiedKFold(n_splits=self.hparams.num_split,
            #            shuffle=True,
            #            random_state=self.hparams.split_seed)
            
            # all_splits = [k for k in kf.split(df, df['diagnosis'])]
            # train_idx, val_idx = all_splits[self.hparams.k_idx]
            # train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            # train = df.iloc[train_idx]
            # val = df.iloc[val_idx]
            
            self.train_data = CXR_dataset(f'{self.hparams.data_root_dir}',train_files, train_cls, train_cobbs, self.hparams.train_transform)
            self.val_data   = CXR_dataset(f'{self.hparams.data_root_dir}',test_files, test_cls, test_cobbs, self.hparams.val_transform)
            # self.test_data  = CXR_dataset(test_df['img_path'].values, test_df['cls'].values, self.hparams.val_transform)
            

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
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.hparams.batch_size, # self.hparams.batch_size, # when PT -> bz else 1 
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
    


if __name__ == "__main__":
    from monai.transforms import (
                            Activations,
                            Activationsd,
                            AsDiscrete,
                            AsDiscreted,
                            ConvertToMultiChannelBasedOnBratsClassesd,
                            Compose,
                            Invertd,
                            LoadImaged,
                            MapTransform,
                            NormalizeIntensityd,
                            Orientationd,
                            RandFlipd,
                            RandScaleIntensityd,
                            RandShiftIntensityd,
                            RandSpatialCropd,
                            CropForegroundd,
                            RandAffined,
                            Resized,
                            Spacingd,
                            EnsureTyped,
                            EnsureChannelFirstd,
                            ScaleIntensityRanged
                        )
    
    train_transform = None
    test_transform = test_transform = Compose([
        LoadImaged(keys='image', image_only=True, ensure_channel_first=True),
        EnsureTyped(keys='image', device=None, track_meta=False),
        Resized(keys='image', spatial_size=[512, 512]),
        ScaleIntensityRanged(keys='image',
                             a_max=255.0, a_min=0., b_max=1, b_min=0, clip=True),
        NormalizeIntensityd(keys='image', subtrahend=0.48833441563848673, divisor=0.24424955053273747),
    ])
    
    

    pl_dataloader = KFold_pl_DataModule(
                                        train_transform=train_transform,
                                        batch_size=1,
                                        num_workers=2,
                                        val_transform=test_transform)

    val_dataloader = pl_dataloader.train_dataloader()


    for idx, batch in enumerate(val_dataloader):
        image, seg_label = batch["image"], batch["y"]
        
        
