# unified all csv file
# split by class labels 

import pandas as pd 
import glob 

from sklearn.model_selection import StratifiedShuffleSplit



# source_dir = '/home/pwrai/userarea/spineTeam/data/encoded'
# csv_list = glob.glob(f'{source_dir}/*')
target_dir = '/home/pwrai/userarea/spineTeam/data'

# df = pd.read_csv(csv_list[0])

# for csv in csv_list[1:]:
#     temp_df = pd.read_csv(csv)
#     df = pd.concat([df, temp_df], axis=0)
    
# df.to_csv(f'{target_dir}/whole_data.csv', index=False) 
df = pd.read_csv('/home/pwrai/userarea/spineTeam/data/C101/info.csv')
skf = StratifiedShuffleSplit(n_splits=1, # default value, since we need spliting only once, it doesn't matter what n_splits value is if x > 0.
                             test_size=0.2,
                             train_size=0.8,
                             random_state=411) # 0411 is ma BD

for i, (train_idx, test_idx) in enumerate(skf.split(df, df['class'])):
    print(f"Fold {i}:")
    print(f"  Train: index={train_idx}, len={len(train_idx)}")
    print(f"  Test:  index={test_idx}, len={len(test_idx)}")
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(train_df.shape, test_df.shape)
    
    train_df.to_csv(f'{target_dir}/train_split_c101.csv', index=False)
    test_df.to_csv(f'{target_dir}/test_split_c101.csv', index=False)
