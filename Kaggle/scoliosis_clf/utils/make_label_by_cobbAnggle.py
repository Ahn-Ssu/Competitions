import pandas as pd
import glob 


source_dir = '/home/pwrai/ktldata/spine/spine1/Cobb/*.csv'
file_list = glob.glob(source_dir)

target_dir = '/home/pwrai/userarea/spineTeam/data'


for csv in file_list:
    
    df = pd.read_csv(csv)
    
    cobbs = df['Cobbs angle'].values
    
    cls_list = []
    diag_list = []
    cls = 0
    diagnosis = None
    for val in cobbs:
        if val < 10:
            cls = 0
            diagnosis='spinalCurve'
        
        if val >= 10 and val < 20:
            cls = 1
            diagnosis='mildScoliosis'
            
        if val >= 20 and val < 40:
            cls = 2
            diagnosis='moderateScoliosis'
        
        if val >= 40:
            cls = 3
            diagnosis='severeScoliosis'
            
        cls_list.append(cls)
        diag_list.append(diagnosis)

    df['class'] = cls_list
    df['diagnosis'] = diag_list
    
    df.to_csv(f'{target_dir}/{csv.split("/")[-1]}', index=False)