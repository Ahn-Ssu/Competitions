import pandas as pd
import glob


train_data_list = glob.glob('/home/pwrai/userarea/spineTeam/data/test_data(4KTL)/train/*/*')


dir_path = []
file_name = []

for one in train_data_list:
    
    splited = one.split('/')
    
    name = splited[-1]
    path = '/'.join(splited[:-1])
    
    dir_path.append(path)
    file_name.append(name)
    

    
df = pd.DataFrame([dir_path, file_name], index=['dir_path', 'file_name'])
df = df.transpose()
df.to_csv('train_files(4KTL_validation).csv', index=None, header=None)
test_data_list = glob.glob('/home/pwrai/userarea/spineTeam/data/test_data(4KTL)/test/*/*')


dir_path = []
file_name = []

for one in test_data_list:
    
    splited = one.split('/')
    
    name = splited[-1]
    path = '/'.join(splited[:-1])

    dir_path.append(path)
    file_name.append(name)

    
df = pd.DataFrame([dir_path, file_name], index=['dir_path', 'file_name'])
df = df.transpose()
df.to_csv('test_files(4KTL_validation).csv', index=None, header=None)