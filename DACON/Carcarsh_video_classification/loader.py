import cv2
import torch
import numpy as np
import pandas as pd

from einops import rearrange
from decord import VideoReader
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, args):
        self.video_path_list = video_path_list
        self.label_list = label_list
        # self.transform = transform
        self.args = args

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
        # path =  f'/root/Competitions/DACON/Carcarsh_video_classification/data/{path[2:]}'

        # vr = VideoReader(path)
        # video = torch.from_numpy(vr.get_batch(range(self.args.video_length)).asnumpy())
        # video = rearrange(video, 't h w c -> c t h w')
        # if self.transform:
        #     video = self.transform(video)

        # # video = rearrange(video, 'c t h w -> t c h w')
        # return video

        frames = []
        path =  f'/root/Competitions/DACON/Carcarsh_video_classification/data/{path[2:]}'
        cap = cv2.VideoCapture(path)
        for _ in range(self.args.video_length):
            _, img = cap.read()
            img = cv2.resize(img, (self.args.img_size, self.args.img_size))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


def label_encoding(df):
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

    labels = df.label.to_numpy()
    encoded = []

    for label in labels:
        encoded.append(enc_lookup[label])
    encoded = pd.DataFrame(np.array(encoded))
    new_features = ['crash','ego','weather','timing']
    encoded.columns = new_features
    enc_df = pd.concat([df, encoded], axis=1)

    return enc_df

def label_decoding(df):

    dec_lookup = {
        (0,0,0,0):0,
        (1,1,1,1):1,
        (1,1,1,2):2,
        (1,1,2,1):3,
        (1,1,2,2):4,
        (1,1,3,1):5,
        (1,1,3,2):6,
        (1,2,1,1):7,
        (1,2,1,2):8,
        (1,2,2,1):9,
        (1,2,2,2):10,
        (1,2,3,1):11,
        (1,2,3,2):12,
    }
    df['crash'] = np.where(df.ego == 0, 0, 1)
    df['weather'] = df['weather'] + 1 
    df['timing'] = df['timing'] + 1 

    df['weather'] = np.where( df.crash == 0, 0 , df.weather)
    df['timing'] = np.where( df.crash == 0, 0 , df.timing)
    
    new_features = ['crash','ego','weather','timing']


    enc_pred = df[new_features].to_numpy()
    decoded = []

    for enc in enc_pred:
        decoded.append(dec_lookup[tuple(enc)])
    
    return pd.DataFrame(np.array(decoded))


def label_shift(enc_df, target):

    num_classes = 3 if target in ['ego','weather'] else 2

    if target in ['weather','timing']:
        df = enc_df[enc_df[target] > 0]
        df[target] = df[target] - 1
    else:
        df = enc_df

    return df, num_classes


    ###########################################################
    ###########################################################

# class VideoDataset(Dataset):
#     def __init__(self, df_for_dataset, transform=None):
#         self.sample_id = df_for_dataset[:,0]
#         self.video_path = df_for_dataset[:,1]
#         self.label = df_for_dataset[:,2]
#         self.label_split = np.array(df_for_dataset[:,3].tolist())
#         self.transform = transform

#     def __len__(self):
#         return len(self.sample_id)

#     def __getitem__(self, idx):
#         sample_id = self.sample_id[idx]
#         video_path = self.video_path[idx]
#         vr = VideoReader(video_path)
#         video = torch.from_numpy(vr.get_batch(range(50)).asnumpy())
#         video = rearrange(video, 't h w c -> c t h w')
#         label = self.label[idx]
#         label_split = self.label_split[idx]
        
#         if self.transform:
#             video = self.transform(video)
#         video = rearrange(video, 'c t h w -> t c h w')

#         sample = {
#             'sample_id':sample_id,
#             'video':video,
#             'label':label,
#             'label_split':label_split
#         }
        
#         return sample