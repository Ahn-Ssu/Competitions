import os, csv
import torch
import pandas as pd

from easydict import EasyDict
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime, timezone, timedelta
from pytorchvideo.transforms.transforms_factory import create_video_transform


import loader 
import networks
import trainer


args = EasyDict()

args.video_length = 50  # 10프레임 * 5초
args.img_size     = 128

args.batch_size   = 4
args.epochs       = 20
args.init_lr      = 3e-4

args.seed = 41

print('efficient, def')
trainer.seed_everything(args.seed)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

savePath = '/root/Competitions/DACON/Carcarsh_video_classification/log'
try:
    if not os.path.exists(savePath):
        os.makedirs(savePath)
except OSError:
    print("Error: Cannot create the directory {}".format(savePath))

train_df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/train.csv')

########## 1. label processing -> label encoding ##########
###########################################################
enc_lookup = {
        -1:[-1,-1,-1,-1],
        0:[0,0,0,0],
        1:[1,1,1,1],
        2:[1,1,1,2],
        3:[1,1,2,1],
        4:[1,1,2,2],
        5:[1,1,3,1],
        6:[1,1,3,2],
        7:[1,2,1,1],
        8:[1,2,1,2],
        9:[1,2,2,1],
        10:[1,2,2,2],
        11:[1,2,3,1],
        12:[1,2,3,2]
    }

labels = train_df.label.to_numpy()
encoded = []

for label in labels:
    encoded.append(enc_lookup[label])
import numpy as np
encoded = pd.DataFrame(np.array(encoded))
new_features = ['crash','ego','weather','timing']
encoded.columns = new_features
enc_df = pd.concat([train_df, encoded], axis=1)


###########################################################
###########################################################
stf_kfold = RepeatedStratifiedKFold(n_splits=5, random_state=args.seed)
target = new_features[2] # To do: ego[1], weather[2], timing[3]

num_classes = 3 if target in ['ego','weather'] else 2

if target in ['weather','timing']:
    df = enc_df[enc_df[target] > 0]
    df[target] = df[target] - 1
else:
    df = enc_df


KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)
today = str(time_record)[:10]

model_name = 'baseLINE'
file_label = f'{savePath}/{today}_LOG__[{target}]_PLAIN_model:{model_name}.csv'

if not os.path.exists(file_label):
      with open(file_label, mode='w') as f:
        myWriter = csv.writer(f)
        myWriter.writerow(['Fold','epoch','trainLoss','valLoss','val_F1','val_Acc'])

archive_path = f'/root/Competitions/DACON/Carcarsh_video_classification/model/{target}/'
try:
    if not os.path.exists(archive_path):
        os.makedirs(archive_path)
except OSError:
    print("Error: Cannot create the directory {}".format(archive_path))

for stage, (train_idx, val_idx) in enumerate(stf_kfold.split(df, df[target])):
    print(f'Current stage of Fold: {stage+1}, target label: {target}')
    print(f'Current stage of Fold: {stage+1}, target label: {target}')

    train = df.iloc[train_idx]
    val   = df.iloc[val_idx]

    #### maybe we need an augmentation method here ###
    ##################################################


    train_dataset = loader.CustomDataset(train['video_path'].values, train[target].values, args=args)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

    val_dataset = loader.CustomDataset(val['video_path'].values, val[target].values, args=args)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2)


    model = networks.BaseModel(num_classes)
    model.name = model_name
    model.eval()
    optimizer = trainer.Apollo(params = model.parameters(), lr = args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

    best_model, logs, best_info = trainer.train(model, optimizer, train_loader, val_loader, scheduler, device, args)

    with open(file_label, mode='a') as f:
        myWriter = csv.writer(f)
        for log in logs:
            myWriter.writerow([stage+1] + log)

    torch.save(best_model.state_dict(), f'{archive_path}{best_model.name}_Fold[{stage}]{best_info}.pth')
    





# torch.save(model.state_dict(), '/root/Competitions/DACON/Carcarsh_video_classification/model/r3d_18(focalLoss, bz 4, epochs 10, Apollo).pth')

# test_df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/test.csv')
# test_dataset = loader.CustomDataset(test_df['video_path'].values, None, args)
# test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)


# preds = learner.inference(model, test_loader, device)

# submit = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/sample_submission.csv')
# submit['label'] = preds
# submit.to_csv('/root/Competitions/DACON/Carcarsh_video_classification/prediction/r3d_18(focalLoss, bz 4, epochs 10, Apollo)_submit.csv', index=False)