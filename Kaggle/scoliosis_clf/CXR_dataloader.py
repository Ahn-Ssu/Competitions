import glob
import pandas as pd
import pytorch_lightning as pl

import torch

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged, AsDiscrete


class CXR_dataset(Dataset):
    def __init__(self, data_dir= None, image_path=None, cls=None, cobbs=None,  transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_path = image_path
        self.cls = cls
        self.cobbs = cobbs
        self.transform = transform
        self.onehot = AsDiscrete(to_onehot=4)

    def __getitem__(self, index):
        return self.get_SCANS(self.image_path[index], self.cls[index], self.cobbs[index])

    def __len__(self):
        return len(self.image_path)
    
    def get_SCANS(self, path:str, cls, cobbs):
        path = path.split('.')[0]
        y = torch.LongTensor(cls)
        class_onehot = self.onehot(y).astype(torch.long)
        path_d = {
            'image': f'{self.data_dir}/{path}.png',
            'y':y,
            'class_onehot':class_onehot,
	        'cobbs':torch.tensor(cobbs, dtype=torch.float)
        }

        try:
            data_d = self.transform(path_d)
        except Exception as err:
            print(err, path_d)
            exit()

        if isinstance(data_d, list):
                data_d = data_d[0]
                
        c, h, w = data_d['image'].size()
        
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
            train_df = pd.read_csv(f'{self.hparams.data_root_dir}/train_split.csv')
            train_files = train_df['File_ID'].values
            train_cls = train_df['class'].values 
            train_cobbs = train_df['Cobbs angle'].values / 50.
            
            
            test_df = pd.read_csv(f'{self.hparams.data_root_dir}/test_split.csv')
            test_files = test_df['File_ID'].values
            test_cls = test_df['class'].values
            test_cobbs = test_df['Cobbs angle'].values / 50.

            # in AI device data, we need the following codes
            # kf = StratifiedKFold(n_splits=self.hparams.num_split,
            #            shuffle=True,
            #            random_state=self.hparams.split_seed)
            
            # all_splits = [k for k in kf.split(df, df['diagnosis'])]
            # train_idx, val_idx = all_splits[self.hparams.k_idx]
            # train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            # train = df.iloc[train_idx]
            # val = df.iloc[val_idx]
            
            self.train_data = CXR_dataset(f'{self.hparams.data_root_dir}/AP',train_files, train_cls, train_cobbs, self.hparams.train_transform)
            self.val_data   = CXR_dataset(f'{self.hparams.data_root_dir}/AP',test_files, test_cls, test_cobbs, self.hparams.val_transform)
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
        
        
