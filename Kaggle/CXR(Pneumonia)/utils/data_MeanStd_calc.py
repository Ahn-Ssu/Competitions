import glob

import numpy as np

from tqdm import tqdm 
from monai.transforms import (Compose, LoadImage, EnsureType, ScaleIntensityRange, NormalizeIntensity)

loader = Compose(
    LoadImage(ensure_channel_first=True),
    EnsureType(device=None, track_meta=False),
    # ScaleIntensityRange(a_min=0., a_max=255., b_min=0., b_max=1., clip=True), TypeError: unsupported operand type(s) for -: 'dict' and 'float'
)

data_path = glob.glob('/root/Competitions/Kaggle/CXR(Pneumonia)/data/chest_xray/train/*/*')


cnt = 0 
x = 0
y = 0 
mean = 0 
var = 0

for path in tqdm(data_path):
    
    data = loader(path)[0]
    # print( data.size())
    c, w, h = data.size()
    
    x += w
    y += h
    data = data.flatten()
    img_mean = np.mean(data)
    img_var  = np.var(data)
    
    L = len(data)
    cnt += L
    mean += L * img_mean
    var += (img_var + img_mean**2) * L    
    
overall_mean = mean / cnt
overall_var = var / cnt - overall_mean**2
overall_var = np.sqrt(overall_var)

norm_offset = 255


print(f'{overall_mean=}')
print(f'{overall_var=}')
print(f'\t {norm_offset=} | norm.ed mean = {overall_mean/norm_offset}')
print(f'\t {norm_offset=} | norm.ed var  = {overall_var/norm_offset}')
print()
print('validity check  validity check  validity check')
print('validity check  validity check  validity check')
print('\t min-max -> [0, 1] -> standardization')


scaler = Compose([
    ScaleIntensityRange(a_min=0., a_max=255., b_min=0., b_max=1., clip=True),
    NormalizeIntensity(subtrahend=overall_mean/norm_offset, divisor=overall_var/norm_offset)
])

cnt = 0 
x = 0
y = 0 
mean = 0 
var = 0

for path in tqdm(data_path):
    
    data = loader(path)[0]
    c, w, h = data.size()
    
    x += w
    y += h
    data = scaler(data)
    data = data.flatten()
    img_mean = np.mean(data)
    img_var  = np.var(data)
    
    L = len(data)
    cnt += L
    mean += L * img_mean
    var += (img_var + img_mean**2) * L
    

standarized_mean = mean / cnt
stndz_var = var / cnt - standarized_mean**2
stndz_var = np.sqrt(stndz_var)

print(f'\tstndz_mean = {round(standarized_mean,4)}')
print(f'\tstndz_var = {round(stndz_var,4)}')