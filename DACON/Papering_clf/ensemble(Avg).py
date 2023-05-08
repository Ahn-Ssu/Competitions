import os
# os.environ["CUDA_VISIBLE_DEVICES"] =""
import glob
import pandas as pd

from easydict import EasyDict
from sklearn import preprocessing
from tqdm import tqdm

from data_loader import *
from lighting import LightningRunner
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from model.models import BaseModel
import numpy as np


args = EasyDict()

args.img_size = 544
args.batch_size = 4


test_transform = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

le = preprocessing.LabelEncoder()

all_img_list = glob.glob('./aug_data/train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])
df['label'] = le.fit_transform(df['label'])

test = pd.read_csv('/root/Competitions/DACON/Papering_clf/aug_data/test.csv')
path = test['img_path'].values
path = [f'/root/Competitions/DACON/Papering_clf/aug_data/test/{file.split("/")[-1]}' for file in path]


test_dataset = CustomDataset(path, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

CV_path = '/root/Competitions/DACON/Papering_clf/lightning_logs/5fold_convNeXt-Large_P'
ckpts = glob.glob(f'{CV_path}/*/checkpoints/*')

inferences = []
DEVICE = 'cuda:0'

for ckpt in ckpts:
    model = BaseModel(19).to(DEVICE)
    pl_runner = LightningRunner.load_from_checkpoint(ckpt, network=model, args=args).to(DEVICE)
    # pl_runner
    pl_runner.eval()

    inference  = []
    for x in tqdm(test_loader):
        x = x.to(DEVICE)

        pred = pl_runner.model(x)
        inference += pred.detach().cpu().numpy().tolist()

        del x 
    
    inferences.append(inference)

inferences = np.mean(np.array(inferences), axis=0)
inferences = np.argmax(inferences, axis=1)

inferences = le.inverse_transform(inferences)
submit = pd.read_csv('/root/Competitions/DACON/Papering_clf/aug_data/sample_submission.csv')
submit['label'] = inferences
submit.to_csv('/root/Competitions/DACON/Papering_clf/prediction/5fold_convNeXt-Large_P-p544.csv', index=False)
    



