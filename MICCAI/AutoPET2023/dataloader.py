import pandas as pd
import pytorch_lightning as pl

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged


class PETCT_dataset(Dataset):
    def __init__(self, image_path=None, diagnosis=None,  transform=None) -> None:
        super().__init__()
        assert image_path and diagnosis

        self.image_path = image_path
        self.diagnosis = diagnosis
        self.transform = transform

    def __getitem__(self, index):
        return self.get_SCANS(self.image_path[index], self.diagnosis[index])

    def __len__(self):
        return len(self.image_path)
    
    def get_SCANS(self, path, diagnosis):
        
        ctres = f'{path}/CTres.nii.gz'
        suv = f'{path}/SUV.nii.gz'
        seg = f'{path}/SEG.nii.gz'

        path_d = {
            'image': (ctres, suv),
            'label': seg
        }

        data_d = LoadImaged(keys=['image','label'])(path_d)

        if self.transform:
            data_d = self.transform(data_d)

        path_d['diagnosis'] = diagnosis

        return data_d
    


class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = '/root/Competitions/MICCAI/AutoPET2023/data/train',
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
            meta_df = pd.read_csv('/root/Competitions/MICCAI/AutoPET2023/data/Metadata-FDG_PET_CT.csv')
            file_paths = meta_df['File Location'].apply(lambda x: '/'.join([self.data_dir] + x.split('/')[2:4])) # Even faster 


            df = pd.DataFrame(columns=['img_path', 'diagnosis'])
            df['img_path'] = file_paths
            df['diagnosis'] = meta_df.diagnosis

            for idx in range(len(meta_df)): assert df.iloc[idx].diagnosis== meta_df.iloc[idx].diagnosis

            le = preprocessing.LabelEncoder()
            df['diagnosis'] = le.fit_transform(df['diagnosis'])

            kf = StratifiedKFold(n_splits=self.hparams.num_split,
                       shuffle=True,
                       random_state=self.hparams.split_seed)
            
            all_splits = [k for k in kf.split(df, df['diagnosis'])]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train = df.iloc[train_idx]
            val = df.iloc[val_idx]
            self.num_cls = len(le.classes_)
            
            self.train_data = PETCT_dataset(train['img_path'].values, train['label'].values, self.hparams.train_transform)
            self.val_data = PETCT_dataset(val['img_path'].values, val['label'].values, self.hparams.val_transform)

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