import glob
import pandas as pd
import pytorch_lightning as pl

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged


class CXR_dataset(Dataset):
    def __init__(self, image_path=None, cls=None,  transform=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.cls = cls
        self.transform = transform

    def __getitem__(self, index):
        return self.get_SCANS(self.image_path[index], self.cls[index])

    def __len__(self):
        return len(self.image_path)
    
    def get_SCANS(self, path:str, cls):
        
        path_d = {
            'image': path,
            'y':cls
        }

        try:
            data_d = self.transform(path_d)
        except Exception as err:
            print(err, path_d)
            exit()

        if isinstance(data_d, list):
                data_d = data_d[0]

        return data_d
    


class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root_dir: str = '/root/Competitions/Kaggle/CXR(Pneumonia)/data/chest_xray',
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
            train_file_paths = glob.glob(f'{self.hparams.data_root_dir}/train/*/*')
            train_df = pd.DataFrame(columns=['img_path', 'cls'])
            train_df['img_path'] = train_file_paths
            train_df['cls'] = train_df['img_path'].apply(lambda x: 0 if x.split('/')[-2] == 'NORMAL' else 1)
            
            # validation data
            val_file_paths = glob.glob(f'{self.hparams.data_root_dir}/val/*/*')
            val_df = pd.DataFrame(columns=['img_path', 'cls'])
            val_df['img_path'] = val_file_paths
            val_df['cls'] = val_df['img_path'].apply(lambda x: 0 if x.split('/')[-2] == 'NORMAL' else 1)
            
            # test data
            test_file_paths = glob.glob(f'{self.hparams.data_root_dir}/test/*/*')
            test_df = pd.DataFrame(columns=['img_path', 'cls'])
            test_df['img_path'] = test_file_paths
            test_df['cls'] = test_df['img_path'].apply(lambda x: 0 if x.split('/')[-2] == 'NORMAL' else 1)

            # in AI device data, we need the following codes
            # kf = StratifiedKFold(n_splits=self.hparams.num_split,
            #            shuffle=True,
            #            random_state=self.hparams.split_seed)
            
            # all_splits = [k for k in kf.split(df, df['diagnosis'])]
            # train_idx, val_idx = all_splits[self.hparams.k_idx]
            # train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            # train = df.iloc[train_idx]
            # val = df.iloc[val_idx]
            
            self.train_data = CXR_dataset(train_df['img_path'].values, train_df['cls'].values, self.hparams.train_transform)
            self.val_data   = CXR_dataset(val_df['img_path'].values, val_df['cls'].values, self.hparams.val_transform)
            self.test_data  = CXR_dataset(test_df['img_path'].values, test_df['cls'].values, self.hparams.val_transform)
            

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
                        )
    
    train_transform = None
    test_transform = test_transform = Compose(
                [
                    EnsureChannelFirstd(keys=["image", "label"]),
                    EnsureTyped(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    # Spacingd(
                    #     keys=["image", "label"],
                    #     pixdim=(1.0, 1.0, 1.0),
                    #     mode=("bilinear", "nearest"),
                    # ),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            ) 
    
    

    pl_dataloader = KFold_pl_DataModule(data_dir='/root/Competitions/MICCAI/AutoPET2023/data/train',
                                        train_transform=train_transform,
                                        batch_size=1,
                                        val_transform=test_transform)

    val_dataloader = pl_dataloader.val_dataloader()


    for idx, batch in enumerate(val_dataloader):
        image, seg_label = batch["image"], batch["label"]
        print(f'{idx}, {image.shape=}, {seg_label.shape=}')
        
        