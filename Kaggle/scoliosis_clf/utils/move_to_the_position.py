import shutil
import os
import pandas as pd

from tqdm import tqdm

def extract_name_N_modify(df, code):
#     df['File Path'] = df['File_ID'].apply(lambda x: f"/home/pwrai/userarea/spineTeam/data/{code}")
#     df['File Name'] = df['File_ID'].apply(lambda x: f"{x.split('/')[-1].split('.')[0]}.png")
    return df['File_ID'].apply(lambda x: f"/home/pwrai/userarea/spineTeam/data/{code}/{x.split('/')[-1].split('.')[0]}.png")

train_c101 = pd.read_csv('/home/pwrai/userarea/spineTeam/data/train_split_c101.csv')
train_c101['File_ID'] = extract_name_N_modify(train_c101, 'C101')
train_c103 = pd.read_csv('/home/pwrai/userarea/spineTeam/data/train_split_c103.csv')
train_c103['File_ID'] = extract_name_N_modify(train_c103, 'C103')

test_c101 = pd.read_csv('/home/pwrai/userarea/spineTeam/data/test_split_c101.csv')
test_c101['File_ID'] = extract_name_N_modify(test_c101, 'C101')
test_c103 = pd.read_csv('/home/pwrai/userarea/spineTeam/data/test_split_c103.csv')
test_c103['File_ID'] = extract_name_N_modify(test_c103, 'C103')


train_df = pd.concat([train_c101, train_c103], axis=0)
test_df = pd.concat([test_c101, test_c103], axis=0)

train_path = '/home/pwrai/userarea/spineTeam/data/test_data(4KTL)/train'
test_path = '/home/pwrai/userarea/spineTeam/data/test_data(4KTL)/test'

for case in tqdm(train_df.values):
    file_path, cobbs, cls, *_ = case
    
    filename = file_path.split('/')[-1]
    diagnosis = 'scoliosis' if cobbs >= 10 else 'normal'
    shutil.copy(file_path, f'{train_path}/{diagnosis}/{filename}')
    
for case in tqdm(test_df.values):
    file_path, cobbs, cls, *_ = case
    
    filename = file_path.split('/')[-1]
    diagnosis = 'scoliosis' if cobbs >= 10 else 'normal'
    shutil.copy(file_path, f'{test_path}/{diagnosis}/{filename}')
    
    
    
    