import glob
import traceback

import numpy as np

from tqdm import tqdm 
from monai.transforms import (Compose, LoadImage, EnsureType, ScaleIntensityRange, NormalizeIntensity)

loader = Compose(
    LoadImage(ensure_channel_first=True),
    EnsureType(device=None, track_meta=False),
    # since we've scaled the data into (0, 255 = uint8), ...
    ScaleIntensityRange(a_min=0., a_max=2**8-1, b_min=0., b_max=1., clip=True),# TypeError: unsupported operand type(s) for -: 'dict' and 'float'
)


data_path = glob.glob('/home/pwrai/userarea/spineTeam/data/AP/*')

cnt = 0 
x = 0
y = 0 
mean = 0 
var = 0

for path in tqdm(data_path):
    try: 
        data = loader(path)[0]
    except Exception as ex:
        print('=====except statements=====')
        print('=====except statements=====')
        err_msg = traceback.format_exc()
        print('source path:', path)
        print('basic infomation---', ex)
        print('detailed')
        print(err_msg)
        continue
    
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



print(f'{overall_mean}')
print(f'{overall_var}')
print()
print('validity check  validity check  validity check')
print('validity check  validity check  validity check')
print('\t min-max -> [0, 1] -> standardization')


scaler = Compose([
    ScaleIntensityRange(a_min=0., a_max=255., b_min=0., b_max=1., clip=True),
    NormalizeIntensity(subtrahend=overall_mean, divisor=overall_var)
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
