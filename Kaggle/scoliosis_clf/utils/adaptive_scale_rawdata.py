import glob
import numpy as np
from tqdm import tqdm
import pydicom
from PIL import Image
import SimpleITK as sitk
import os 
import pandas as pd


def scale_img(arr, offset):
	# for safty
	max_val = offset - 1
	arr = np.where( arr>max_val, max_val, arr)
	scaled = ((arr ) / (max_val) *255.).astype(np.uint8)
	if len(scaled.shape) == 3:
		scaled = scaled[0]
	return scaled


filtered_dtype = []
filtered_offsets = []
filtered_path = []
filtered_angle = []
filtered_class = []
filtered_diagnosis = []

csv_path = '/home/pwrai/userarea/spineTeam/data/whole_data.csv'
df = pd.read_csv(csv_path, index_col=False)

file_path = df["File_ID"].values
cobb_angle = df['Cobbs angle'].values
scoliosis_calss = df['class'].values
diagnosis = df['diagnosis'].values

save_dir = '/home/pwrai/userarea/spineTeam/data/C106'
for idx in tqdm(range(len(file_path))):
	file = file_path[idx]
	path = f'/home/pwrai/ktldata/spine/spine1/DCM/{file}'
	
	if not 'C106' in path:
		continue
        
	dcm = sitk.ReadImage(path)
	arr = sitk.GetArrayFromImage(dcm)

	if str(arr.dtype) == 'float64' or str(arr.dtype) == 'float':
		continue

# 	if not '0038' in path:
# 		continue        
        
# 	bit8_norm = scale_img(arr, 2**8) / 255
# 	bit10_norm = scale_img(arr, 2**10) / 255
# 	bit12_norm = scale_img(arr, 2**12) / 255
# 	bit1210_norm = scale_img(arr, 2**12 + 2 **10) / 255
# 	bit14_norm = scale_img(arr, 2**14) / 255
    
	for offset in [2**8, 2**10, 2**12, 2**12 + 2**10, 2**14]: 
		uint8_img = scale_img(arr, offset)
		mean = np.mean(uint8_img) / 255.
		print(path, offset, round(mean,4))
# 		img = Image.fromarray(uint8_img)
# 		filename = path.split('/')[-1].split('.')[0]
# 		save_name = f'{save_dir}/{filename}_{offset}.png'
# 		img.save(save_name)
		if mean > 0.5:
			continue
		else:
			
			break
	if mean < 0.35:
			continue
	
        
# 	print(f'8    mean - {np.mean(bit8_norm)}')
# 	print(f'10   mean - {np.mean(bit10_norm)}')
# 	print(f'12   mean - {np.mean(bit12_norm)}')
# 	print(f'1210 mean - {np.mean(bit1210_norm)}')
# 	print(f'14   mean - {np.mean(bit14_norm)}\n')
	
	img = Image.fromarray(uint8_img)
	filename = path.split('/')[-1].split('.')[0]
	save_name = f'{save_dir}/{filename}.png'
	img.save(save_name)
 
			
	filtered_dtype.append(str(arr.dtype)) 
	filtered_offsets.append(offset)
	filtered_path.append(file)
	filtered_angle.append(cobb_angle[idx])
	filtered_class.append(scoliosis_calss[idx])
	filtered_diagnosis.append(diagnosis[idx])
	


df = pd.DataFrame([filtered_path, filtered_angle, filtered_class,  filtered_diagnosis, filtered_dtype, filtered_offsets], 
                  index=['File_ID', 'Cobbs angle','class','diagnosis','dtype','offset'])
df = df.transpose()


df.to_csv(f'{save_dir}/info.csv', index=False)

