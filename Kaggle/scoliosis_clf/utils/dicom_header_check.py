import traceback
import pydicom
import SimpleITK as sitk

import numpy as np
import glob

from tqdm import tqdm

from PIL import Image
import pandas as pd 

path_list = glob.glob('/home/pwrai/ktldata/spine/spine1/DCM/*')


err_list = []

saved_name = []
bits_list = []
s_list = []
b_list = []
center_list = []
width_list = []
photometric_list = []

save_dir = '/home/pwrai/userarea/spineTeam/data/val_header'
for dicom_path in tqdm(path_list):
    dicom_err = False
    
    slope = None
    intercept = None
    
    try:
        dicom = pydicom.read_file(dicom_path)
        
        meta_info = dicom.dir()
        if 'RescaleSlope' in meta_info and 'RescaleSlope' in meta_info: 
            slope = dicom.RescaleSlope
            intercept = dicom.RescaleIntercept
        
        window_center = dicom.WindowCenter
        window_width = dicom.WindowWidth
        
        photometric = dicom.PhotometricInterpretation
        
        bits_stored = dicom.BitsStored
        representation = dicom.PixelRepresentation 
        
        array = dicom.pixel_array
        
    except Exception as ex:
        print('pydicom lib, err')
#         print('err msg: \n', traceback.format_exc())
        print(ex)
        print(dicom_path)
        dicom_err = True
        
    try:
        if dicom_err:
            
            dicom = sitk.ReadImage(dicom_path)
            
            meta_info = dicom.GetMetaDataKeys()
            
            if '0028|1053' in meta_info and '0028|1052' in meta_info:
                slope = dicom.GetMetaData('0028|1053')
                intercept = dicom.GetMetaData('0028|1052')
            
            window_center = dicom.GetMetaData('0028|1050')
            window_width = dicom.GetMetaData('0028|1051')
            
            photometric = dicom.GetMetaData('0028|0004')
            
            bits_stored = dicom.GetMetaData('0028|0101')
            representation = dicom.GetMetaData('0028|0103')
            
            array = sitk.GetArrayFromImage(dicom)
            
    except Exception as ex:
        print('simpleitk lib, err')
#         print('err msg: \n', traceback.format_exc())
        print(ex)
        print(dicom_path)
        err_list.append(dicom_path)
        continue
        
    
    if not slope ==None and not intercept == None:
        array = slope * array + intercept
    
    
    min_val = window_center - int(window_width/2.0)
    max_val = window_center + int(window_width/2.0)
    array = np.clip(array, a_min=min_val, a_max=max_val)
    
    if photometric == 'MONOCHROME1':
        array = 1.0 - array 
        
    array = array / 2**bits_stored

    img = (array * 255.).astype(np.uint8)
    if len(img.shape) == 3:
        img = img[0]
        
    img = Image.fromarray(img)
    filename = dicom_path.split('/')[-1].split('.')[0]
    save_name = f'{save_dir}/{filename}.png'
    img.save(save_name)
    
    saved_name.append(save_name)
    bits_list.append(bits_stored)
    s_list.append(slope)
    b_list.append(intercept)
    center_list.append(window_center)
    width_list.append(window_width)
    photometric_list.append(photometric)
    
    
df = pd.DataFrame([saved_name, bits_list, s_list, b_list, center_list, width_list, photometric_list], index=['File_ID','bits_stored','slope','intercept','win_center','win_width','photometric'])
df = df.transpose()

df.to_csv(f'{save_dir}/info.csv', index=False)
        
    
        
    
    
    
        
    
        
