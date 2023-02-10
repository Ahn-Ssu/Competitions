import torch
import pandas as pd

from torch.utils.data import DataLoader

import loader 
import model
import learner


CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':256,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':4,
    'SEED':41
}

learner.seed_everything(CFG['SEED'])
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

train_df = loader.load_MetaData('/root/Competitions/DACON/Carcarsh_video_classification/data/train.csv')
train, val = loader.data_split(train_df, seed=CFG['SEED'])

train_dataset = loader.CustomDataset(train['video_path'].values, train['label'].values, CFG=CFG)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = loader.CustomDataset(val['video_path'].values, val['label'].values, CFG=CFG)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


# model = model.efficientNet3D()
model = model.BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

infer_model = learner.train(model, optimizer, train_loader, val_loader, scheduler, device, CFG)


test_df = loader.load_MetaData('/root/Competitions/DACON/Carcarsh_video_classification/data/test.csv')
test_dataset = loader.CustomDataset(test_df['video_path'].values, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


preds = learner.inference(model, test_loader, device)


submit = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/sample_submission.csv')
submit['label'] = preds
submit.to_csv('/root/Competitions/DACON/Carcarsh_video_classification/prediction/efficientNet3d(x2 resolution)_submit.csv', index=False)