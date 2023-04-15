from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import torch


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        # image1 = train_transform_1(image=image)['image']
        # image2 = train_transform_2(image=image)['image']
        # image3 = train_transform_3(image=image)['image']

        # print(image1)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.label_list is not None:
            label = self.label_list[index]
            # print(label)
            # image = torch.stack((image,image1,image2,image3))
            # return image, label.repeat(4, 0)
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)