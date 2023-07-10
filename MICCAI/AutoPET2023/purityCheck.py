import glob 
from monai.transforms import LoadImaged
from tqdm import tqdm
data_dir='/root/Competitions/MICCAI/AutoPET2023/data/train'


scan_cases = glob.glob(f'{data_dir}/*/*')
all_keys = ['ct','pet','label']
loader =LoadImaged(keys=all_keys,
                   ensure_channel_first=True)

for path in tqdm(scan_cases):
    ctres = f'{path}/CTres.nii.gz'
    suv = f'{path}/SUV.nii.gz'
    seg = f'{path}/SEG.nii.gz'

    path_d = {
        'ct': ctres,
        'pet': suv,
        'label': seg,
    }

    try:
        data_d = loader(path_d)
    except:
        print('err occured')
        print(f'{path_d=}')
        continue
    
    del data_d


