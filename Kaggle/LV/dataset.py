import os
import numpy as np
import pickle
import torch
import random
from pathlib import Path
import torchio as tio
import os

## Set deterministic randomness
random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_dataset(data_dir, transform=tio.Compose([])):
    files = os.listdir(data_dir)
    subjects = []
    for file in files:
        data_index = pickle.load(open(f"{data_dir}/{file}", 'rb'))

        mri = data_index['mri'][np.newaxis,:,:,:].astype(np.float32)

        label = data_index['label'][np.newaxis,:,:,:].astype(np.float32)

        table = data_index['table']
        table_label = table[-5:].astype(np.float32) #SNSB
        table_input = table[1:-5].astype(np.float32) # The others, except ID,

        subject = tio.Subject(
            {                
                'mri':tio.ScalarImage(tensor=torch.from_numpy(mri)),
                'label':tio.LabelMap(tensor=torch.from_numpy(label)),
                'table_input':torch.from_numpy(table_input),
                'table_label':torch.from_numpy(table_label),
            }
        )
        subjects.append(subject)
    return tio.SubjectsDataset(subjects, load_getitem=False, transform=transform)  