import os
import numpy as np
import pandas as pd
import nibabel as nib
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

from torch.utils.data import Dataset
from monai.transforms import LoadImaged, MapTransform

class MRI_dataset(Dataset):
    def __init__(self, train_DIR_PATH=None, ext=None, transform=None, is_fold=False) -> None:
        super().__init__()
        assert train_DIR_PATH and ext, "pass a train data path and ext into 'train_DIR_PATH, 'ext"

        self.ext = ext
        self.train_SCAN_PATH = []
        self.transform = transform

        if is_fold:
            folds = [os.path.join(train_DIR_PATH, fold) for fold in os.listdir(train_DIR_PATH) if os.path.isdir(os.path.join(train_DIR_PATH, fold))]
            for fold in folds:
                self.train_SCAN_PATH += [os.path.join(fold, sample_path) for sample_path in os.listdir(fold)]
        else:
            self.train_SCAN_PATH = [os.path.join(train_DIR_PATH, repo) for repo in os.listdir(train_DIR_PATH) if os.path.isdir(train_DIR_PATH)]

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

        
        
        if True in torch.isnan(data_d['image']):
            print(f'image nan check TRUE, path:[{img}]')

        if True in torch.isnan(data_d['label']):
            print(f'label nan check TRUE, path:[{img}]')


        return data_d

# 전체를 다 고려한채로 training 을 해야한다는 생각이 있었는데, 
# 그거 촬영된 사람 별로 다 다를 테니까 그냥 샘플마다 하는게..
# avg_3t = np.average(np.average(np.average(nii_data, axis=0),axis=0), axis=0)
# std_3t = np.std(np.std(np.std(nii_data, axis=0),axis=0), axis=0)
# print(avg_3t)

# nii_data[nii_data == 0] = np.nan
# nanstd_3t = np.nanstd(np.nanstd(np.nanstd(nii_data, axis=0),axis=0), axis=0)
# nanmean_3t = np.nanmean(np.nanmean(np.nanmean(nii_data, axis=0),axis=0), axis=0)
# print(nanmean_3t)
import torch
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

if __name__ == "__main__":
    import sys
    from monai.transforms import (
            Activations,
            Activationsd,
            AsDiscrete,
            AsDiscreted,
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
            Spacingd,
            EnsureTyped,
            EnsureChannelFirstd,
        )
    
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together (into image key)
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    print('=='*40)
    print(f'{sys.argv[0]} TEST '*3)
    print(f'{sys.argv[0]} TEST '*3)
    print()

    path = './data/train'
    print(f'\ttest path : [{path}]')

    myData = MRI_dataset(path, 'nii', train_transform)

    scans = myData[0]
    

    print(f'\treturned scans : [{scans.keys()}]')
    print(f'\treturned scan shape:')
    print(f'\t\timage: [{scans["image"].shape}]')
    print(f'\t\tlabel: [{scans["label"].shape}]')

    
    print()
    print(f'{sys.argv[0]} TEST DONE! '*3)
    print(f'{sys.argv[0]} TEST DONE! '*3)
    print('=='*40)







    

