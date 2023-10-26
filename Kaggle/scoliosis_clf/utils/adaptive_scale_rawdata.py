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
	max_val = offset -1
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

save_dir = '/home/pwrai/userarea/spineTeam/data/AP'
for idx in enumerate(len(file_path)):
    
	path = file_path[idx]
	path = f'/home/pwrai/ktldata/spine/spine1/DCM/{path}'
	
	dcm = sitk.ReadImage(path)
	arr = sitk.GetArrayFromImage(dcm)

	if str(arr.dtype) == 'float64' or str(arr.dtype) == 'float':
		continue

	min_val = 0
 
	if 'C101' in path:
        # 5119 data
		c101_5199 = ['C101_Spine_0330_20190108_1', 'C101_Spine_0149_20190102_1', 'C101_Spine_0084_20190325_1', 'C101_Spine_0081_20191001_1','C101_Spine_0037_20191223_1','C101_Spine_0029_20190718_1']
		
		is_in = False
		for label in c101_5199:
			if path in label:
				is_in = True
				break
		
		offset = 2**12 + 2**10 if is_in else 2**12

	if 'C104' in path or 'C106':
		offset = 2**12


	if 'C103' in path or 'C105' in path or 'C301' in path:

		# manually checking data distribution
		for offset_bit in [2**10, 2**12, 2**12+2**10]:
			bins = np.arange(0, 2**14, offset_bit)
			hist, bins = np.histogram(arr, bins)

			bit_flag = False # for

			for idx, (h, b) in enumerate(zip(hist, bins)):
				if idx == 1 and h < 2**10:
					bit_flag = True
					break

			if bit_flag:
				break

	scaled = scale_img(arr, min_val, offset)
	img = Image.fromarray(scaled)

	filename = path.split('/')[-1].split('.')[0]
	save_name = f'{save_dir}/{filename}.png'
	img.save(save_name)
 
			
	filtered_dtype.append(str(arr.dtype)) 
	filtered_offsets.append(offset)
	filtered_path.append(path)
	filtered_angle.append(cobb_angle[idx])
	filtered_class.append(scoliosis_calss[idx])
	filtered_diagnosis.append(diagnosis[idx])


df = pd.DataFrame([filtered_path, filtered_angle, filtered_class,  filtered_diagnosis, filtered_dtype, filtered_offsets], 
                  columns=['File_ID', 'Cobbs angle','class','diagnosis','dtype','offset'])


df.to_csv(csv_path, index=False)

