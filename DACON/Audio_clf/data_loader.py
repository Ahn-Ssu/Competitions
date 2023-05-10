from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

class CustomDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53") #"facebook/wav2vec2-large-xlsr-53" or 'facebook/wav2vec2-base'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_values = self.processor(
            self.x[idx],         # sound_array,
            sampling_rate=16000, # 16000 Hz
            padding=True,
            return_tensors="pt").input_values
        
        if self.y is not None:
            return input_values.squeeze(), self.y[idx]
        else:
            return input_values.squeeze()