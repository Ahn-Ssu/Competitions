import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_MetaData(path:str=None):
    assert path
    df = pd.read_csv('/root/Competitions/DACON/Carcarsh_video_classification/data/train.csv')

    return df

def data_split(df, seed):

    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=seed)

    return train, val

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, CFG):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.CFG = CFG
        
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self, path):
        frames = []
        path =  f'/root/Competitions/DACON/Carcarsh_video_classification/data/{path[2:]}'
        cap = cv2.VideoCapture(path)
        for _ in range(self.CFG['VIDEO_LENGTH']):
            _, img = cap.read()
            img = cv2.resize(img, (self.CFG['IMG_SIZE'], self.CFG['IMG_SIZE']))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
