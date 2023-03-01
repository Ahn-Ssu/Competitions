import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, args):
        self.video_path_list = video_path_list
        self.label_list = label_list
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
        frames = []
        path =  f'/root/Competitions/DACON/Carcarsh_video_classification/data/{path[2:]}'
        cap = cv2.VideoCapture(path)
        for _ in range(self.args.video_length):
            _, img = cap.read()
            img = cv2.resize(img, (self.args.img_size, self.args.img_size))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)



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