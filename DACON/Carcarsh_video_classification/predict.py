import os
import torch 
import numpy as np
import pandas as pd 

from easydict import EasyDict
from torch.utils.data import DataLoader

import trainer
import loader
import networks

args = EasyDict()

args.video_length = 50
args.img_size = 128

args.batch_size = 4

args.seed = 41


trainer.seed_everything(args.seed)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

weight_path1 = '/root/Competitions/DACON/Carcarsh_video_classification/model/ego/r3d_18--2023-03-03[ego]_Fold[3]INFO_epoch39, tarinLoss0.001, valLoss0.003, valF10.987, valAcc0.993.pth'
# weight_path2 = '/root/Competitions/DACON/Carcarsh_video_classification/model/weather/i3d_r50--2023-03-03[weather]_Fold[1]INFO_epoch39, tarinLoss0.004, valLoss0.117, valF10.78, valAcc0.88.pth'
weight_path3 = '/root/Competitions/DACON/Carcarsh_video_classification/model/timing/r3d_18--2023-03-03[timing]_Fold[24]INFO_epoch5, tarinLoss0.007, valLoss0.028, valF10.974, valAcc0.989.pth'
weight_paths = [weight_path1, weight_path3]
# /root/Competitions/DACON/Carcarsh_video_classification/model/weather

test_df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/test.csv')
submission_df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/sample_submission.csv')


# enc_df = loader.label_encoding(test_df)

model_names = ['r3d_18','r3d_18'] # 'r3d_18', 'r2plus1d_18', 'i3d_r50'
targets = ['ego','timing']

path = '/root/Competitions/DACON/Carcarsh_video_classification/model/weather'
weatherz = os.listdir(path)

for idx in range(len(weatherz)):
    weatherz[idx] = f'{path}/{weatherz[idx]}'

weather_pred = []
for weight_path in weatherz:
    print(weight_path)
    target = 'weather'
    num_classes = 3
    args.model_name = weight_path.split('/')[-1].split('--')[0]

    if args.model_name == 'i3d_r50':
        args.img_size = 196
    else:
        args.img_size = 128

    test_dataset = loader.CustomDataset(test_df['video_path'].values, None, args=args)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2)

    try:
        model = networks.BaseModel(num_classes, args=args)
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except:
        print()
        print(f'err with {weight_path}')
        print()
        continue
    model.eval()

    preds = trainer.inference(model=model, test_loader=test_loader, device=device)
    weather_pred.append(preds)

weather_pred = np.array(weather_pred)

weather_pred = np.average(weather_pred, axis=0)

weather_pred = np.where(weather_pred > 1.5, 2, np.where(weather_pred > 0.5, 1, 0))
test_df['weather'] = weather_pred



for target, weight_path in zip(targets, weight_paths):
    print(f'now target:[{target}]')
    num_classes = 3 if target in ['ego','weather'] else 2
    args.model_name = weight_path.split('/')[-1].split('--')[0]

    if args.model_name == 'i3d_r50':
        args.img_size = 196
    else:
        args.img_size = 128

    test_dataset = loader.CustomDataset(test_df['video_path'].values, None, args=args)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2)


    model = networks.BaseModel(num_classes, args=args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    preds = trainer.inference(model=model, test_loader=test_loader, device=device)
    test_df[target] = preds


print('pred done')


prediction = loader.label_decoding(test_df)
    
submission_df['label'] = prediction
submission_df.to_csv('/root/Competitions/DACON/Carcarsh_video_classification/prediction/2Top_weather(Avg, all)_submit.csv', index=False)