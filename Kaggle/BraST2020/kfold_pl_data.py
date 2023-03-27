import os
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged
from sklearn.model_selection import KFold

class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = None,
                 ext: str=None,
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

    def setup(self, stage: str) -> None:
        if not self.train_data and not self.val_data:
            train_data_paths = [os.path.join(self.hparams.data_dir, repo) for repo in os.listdir(self.hparams.data_dir) if os.path.isdir(self.hparams.data_dir)]

            kf = KFold(n_splits=self.hparams.num_split,
                       shuffle=True,
                       random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(train_data_paths)]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train_list = [train_data_paths[i] for i in train_idx]
            val_list = [train_data_paths[i] for i in val_idx]

            self.train_data = MRI_dataset(DIR_PATH=train_list, ext=self.hparams.ext, transform=self.hparams.train_transform)
            self.val_data = MRI_dataset(DIR_PATH=val_list, ext=self.hparams.ext, transform=self.hparams.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)




class MRI_dataset(Dataset):
    def __init__(self, DIR_PATH=None, ext=None, transform=None, is_fold=False) -> None:
        super().__init__()
        assert DIR_PATH and ext, "pass a train data path and ext into 'DIR_PATH, 'ext"

        self.ext = ext
        self.train_SCAN_PATH = DIR_PATH
        self.transform = transform

    def __getitem__(self, index):
        return self.get_SCANS(self.train_SCAN_PATH[index])

    def __len__(self):
        return len(self.train_SCAN_PATH)
    
    def get_SCANS(self, path):

        img = path.split('/')[-1]
         
        

        seg   = os.path.join(path,f'{img}_seg.{self.ext}')
        flair = os.path.join(path,f'{img}_flair.{self.ext}')
        t1    = os.path.join(path,f'{img}_t1.{self.ext}')
        t1ce  = os.path.join(path,f'{img}_t1ce.{self.ext}')
        t2    = os.path.join(path,f'{img}_t2.{self.ext}')


        data_pathd = {
            'image': (t1, t1ce, t2, flair),
            'label': seg
        }

        data_d = LoadImaged(keys=["image", "label"])(data_pathd)

        if self.transform:
            data_d = self.transform(data_d)

        return data_d
            
        

        