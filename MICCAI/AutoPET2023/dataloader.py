import pandas as pd
import pytorch_lightning as pl

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged


class PETCT_dataset(Dataset):
    def __init__(self, image_path=None, diagnosis=None,  transform=None, negative_val=None) -> None:
        super().__init__()
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
            'ct': ctres,
            'pet': suv,
            'label': seg,
            'diagnosis': diagnosis
        }

        # data_d = LoadImaged(keys=['image','label'])(path_d)
        try:
            data_d = self.transform(path_d)
        except:
            print("RuntimeError: applying transform <monai.transforms.io.dictionary.LoadImaged object at 0x7fd968ab4310>")
            print("same err occur")
            print(path_d)
            exit()

        if isinstance(data_d, list):
                data_d = data_d[0]
            
                

        # if data_d["label"].shape[-1] > 500:
        #     print(path)
        # path_d['diagnosis'] = diagnosis
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
        persistent_workers = True if num_workers > 0 else False
        self.save_hyperparameters(logger=False)

        self.train_data = None
        self.val_data = None
        self.num_cls = 0

        self.setup()

    def setup(self, stage=None) -> None:
        if not self.train_data and not self.val_data:
            meta_df = pd.read_csv('/root/Competitions/MICCAI/AutoPET2023/data/Metadata-FDG_PET_CT.csv')
            meta_df = meta_df[meta_df['SOP Class Name'] == 'Segmentation Storage'] # to remove the redundancy path
            file_paths = meta_df['File Location'].apply(lambda x: '/'.join([self.hparams.data_dir] + x.split('/')[2:4])) # Even faster 

            df = pd.DataFrame(columns=['img_path', 'diagnosis'])
            df['img_path'] = file_paths
            df['diagnosis'] = meta_df.diagnosis

            # df = df[df['diagnosis'] != 'NEGATIVE']

            # for idx in range(len(meta_df)): assert df.iloc[idx].diagnosis== meta_df.iloc[idx].diagnosis

            le = preprocessing.LabelEncoder()
            df['diagnosis'] = le.fit_transform(df['diagnosis'])
            NEGATIVE_VAL = [3] # le.transform(['NEGATIVE'])

            kf = StratifiedKFold(n_splits=self.hparams.num_split,
                       shuffle=True,
                       random_state=self.hparams.split_seed)
            
            all_splits = [k for k in kf.split(df, df['diagnosis'])]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train = df.iloc[train_idx]
            val = df.iloc[val_idx]
            self.num_cls = len(le.classes_)
            
            self.train_data = PETCT_dataset(train['img_path'].values, train['diagnosis'].values, self.hparams.train_transform, NEGATIVE_VAL)
            self.val_data = PETCT_dataset(val['img_path'].values, val['diagnosis'].values, self.hparams.val_transform)

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
                          batch_size=1, # self.hparams.batch_size, # when PT -> bz else 1 
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
        
        