import os, csv
import torch
import pandas as pd

from easydict import EasyDict
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime, timezone, timedelta
from pytorchvideo.transforms.transforms_factory import create_video_transform
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


import loader 
import networks
import trainer


args = EasyDict()

args.video_length = 50  # 10프레임 * 5초
args.img_size     = 196

args.batch_size   = 4
args.epochs       = 40
args.init_lr      = 5e-5

args.seed = 41


trainer.seed_everything(args.seed)
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

savePath = '/root/Competitions/DACON/Carcarsh_video_classification/log'
try:
    if not os.path.exists(savePath):
        os.makedirs(savePath)
except OSError:
    print("Error: Cannot create the directory {}".format(savePath))

train_df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/train.csv')
enc_df = loader.label_encoding(train_df)

stf_kfold = RepeatedStratifiedKFold(n_splits=5, random_state=args.seed)

# To do: ego[1], weather[2], timing[3]
target = 'ego' 
target = 'weather'
target = 'timing'

df, num_classes = loader.label_shift(enc_df, target)


KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)
today = str(time_record)[:10]

model_name = 'i3d_r50'
file_label = f'{savePath}/{today}_LOG__[{target}]_ReduceLROnPlateau, model-[{model_name}], lr-[{args.init_lr}], img-[{args.img_size}].csv'

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
    # scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=15, cycle_mult=1, max_lr=3e-4, min_lr=1e-5, warmup_steps=3, gamma=0.8)

    best_model, logs, best_info = trainer.train(model, optimizer, train_loader, val_loader, scheduler, device, args)

    with open(file_label, mode='a') as f:
        myWriter = csv.writer(f)
        for log in logs:
            myWriter.writerow([stage+1] + log)

    torch.save(best_model.state_dict(), f'{archive_path}{best_model.name}--{today}[{target}]_Fold[{stage}]{best_info}.pth')
    





# torch.save(model.state_dict(), '/root/Competitions/DACON/Carcarsh_video_classification/model/r3d_18(focalLoss, bz 4, epochs 10, Apollo).pth')

# test_df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/test.csv')
# test_dataset = loader.CustomDataset(test_df['video_path'].values, None, args)
# test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)


# preds = learner.inference(model, test_loader, device)

# submit = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/sample_submission.csv')
# submit['label'] = preds
# submit.to_csv('/root/Competitions/DACON/Carcarsh_video_classification/prediction/r3d_18(focalLoss, bz 4, epochs 10, Apollo)_submit.csv', index=False)