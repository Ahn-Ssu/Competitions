import os
import numpy as np
import pandas as pd
import nibabel as nib

from torch.utils.data import Dataset

class MRI_dataset(Dataset):
    def __init__(self, train_DIR_PATH=None, ext=None) -> None:
        super().__init__()
        assert train_DIR_PATH and ext, "pass a train data path and ext into 'train_DIR_PATH, 'ext"

        self.ext = ext
        self.train_SCAN_PATH = []

        folds = [os.path.join(train_DIR_PATH, fold) for fold in os.listdir(train_DIR_PATH) if os.path.isdir(os.path.join(train_DIR_PATH, fold))]
        for fold in folds:
            self.train_SCAN_PATH += [os.path.join(fold, sample_path) for sample_path in os.listdir(fold)]

    def __getitem__(self, index):
        return self.get_SCANS(self.train_SCAN_PATH[index])

    def __len__(self):
        return len(self.train_SCAN_PATH)
    
    def get_SCANS(self, path):

        img = path.split('/')[-1]

        seg   = nib.load(os.path.join(path,f'{img}_seg.{self.ext}'))
        flair = nib.load(os.path.join(path,f'{img}_flair.{self.ext}'))
        t1    = nib.load(os.path.join(path,f'{img}_t1.{self.ext}'))
        t1ce  = nib.load(os.path.join(path,f'{img}_t1ce.{self.ext}'))
        t2    = nib.load(os.path.join(path,f'{img}_t2.{self.ext}'))

        seg   = self.norm(seg, seg=True)
        flair = self.norm(flair)
        t1    = self.norm(t1)
        t1ce  = self.norm(t1ce)
        t2    = self.norm(t2)

        return seg, flair, t1, t1ce, t2

    
    def norm(self, nii_img, seg=False):

        img = nii_img.get_fdata()

        if seg: 
            # Make the label set from [0,1,2,4] to [0,1,2,3]
            img = np.array(img, dtype= np.int32)
            img[img==4] = 3
        else:
            # z-score norm per img
            img = np.array(img, dtype=np.float32)
            fore_mask = img != 0
            img[fore_mask] = (img[fore_mask] -  np.mean(img[fore_mask])) / np.std(img[fore_mask])
        
        return img

# 전체를 다 고려한채로 training 을 해야한다는 생각이 있었는데, 
# 그거 촬영된 사람 별로 다 다를 테니까 그냥 샘플마다 하는게..
# avg_3t = np.average(np.average(np.average(nii_data, axis=0),axis=0), axis=0)
# std_3t = np.std(np.std(np.std(nii_data, axis=0),axis=0), axis=0)
# print(avg_3t)

# nii_data[nii_data == 0] = np.nan
# nanstd_3t = np.nanstd(np.nanstd(np.nanstd(nii_data, axis=0),axis=0), axis=0)
# nanmean_3t = np.nanmean(np.nanmean(np.nanmean(nii_data, axis=0),axis=0), axis=0)
# print(nanmean_3t)

if __name__ == "__main__":
    import sys
    print('=='*40)
    print(f'{sys.argv[0]} TEST '*3)
    print(f'{sys.argv[0]} TEST '*3)
    print()

    path = './data/train'
    print(f'\ttest path : [{path}]')

    myData = MRI_dataset(path, 'nii')

    scans = myData[0]

    print(f'\treturned scans number: [{len(scans)}]')
    print(f'\treturned scan shape:')
    for idx, scan in enumerate(scans):
        print(f'\t\tidx: [{idx}], shape: {scan.shape}')

    
    print()
    print(f'{sys.argv[0]} TEST DONE! '*3)
    print(f'{sys.argv[0]} TEST DONE! '*3)
    print('=='*40)







    

