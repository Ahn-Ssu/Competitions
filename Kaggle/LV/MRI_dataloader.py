import glob

import numpy as np
from sklearn.model_selection import KFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import traceback
import pickle

class MRI_dataset(Dataset):
    def __init__(self, scan_IDs=None, transform=None) -> None:
        super().__init__()
        assert transform

        self.scan_IDs = scan_IDs
        self.transform = transform

    def __getitem__(self, index):
        return self.get_SCANS(self.scan_IDs[index])

    def __len__(self):
        return len(self.scan_IDs)
    
    def get_SCANS(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    
        image, label = data['t1'], data['prep_aseg']


        d = {'image':image,
            'label':label}
        
        if self.transform:
            d = self.transform(d)
        
        return d
    


class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
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
            scan_list = glob.glob('/root/data/lv_seg/*/*/RAS_t1+aseg.pickle')
            
            kf = KFold(n_splits=self.hparams.num_split)
            all_splits = [k for k in kf.split(scan_list)]
            
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            
            train_scans = np.array([scan_list[idx] for idx in train_idx]) # 130
            val_scans = np.array([scan_list[idx] for idx in val_idx]) # 33
            # train_scans = scan_list[train_idx]
            # val_scans = scan_list[val_idx]
            
            # print(val_scans) e.g. '/root/snsb/data/mri/DUIH_0015'
            
            self.train_data = MRI_dataset(train_scans, self.hparams.train_transform)
            self.val_data   = MRI_dataset(val_scans, self.hparams.val_transform)
            

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
                          num_workers=self.hparams.num_workers//4,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.hparams.batch_size, # self.hparams.batch_size, # when PT -> bz else 1 
                          shuffle=False,
                          num_workers=self.hparams.num_workers//4,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
    


if __name__ == "__main__":
    from utils.transform_generator import MONAI_transformerd
    from monai import transforms
    import torch
    
    
    path = '/root/data/lv_seg/ADNI3000_AD/I1003730_135_S_6389/RAS_t1+aseg.pickle'

    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    image, label = data['t1'], data['prep_aseg']

    print(image.min(), image.max(), label.shape)

    d = {'image':image,
        'label':label}
    t = transforms.Compose([
        transforms.EnsureChannelFirstd(keys=['image','label'], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=['label'], dtype=torch.long),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        # transforms.CropForegroundd(keys=['image','label'], source_key=['image']),
        transforms.AsDiscreted(keys=['label'], to_onehot=20),
        transforms.CenterSpatialCropd(keys=["image", 'label'], roi_size=(224, 224, 224)),
    ])

    d = t(d)
    print(d['image'].shape, d['label'].shape)