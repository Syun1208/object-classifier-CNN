import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class Image_Loader(Dataset):
    def __init__(self, root_path='train.csv', image_size=[64, 64], transforms_data=True):
        
        self.data_path = pd.read_csv(root_path)
        self.image_size = image_size
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # read the images from image path
        image_path = os.path.join(self.data_path.iloc[idx, 0])
        image = Image.open(image_path)
        
        # read label
        label_cross = self.data_path.iloc[idx, 1]

        if self.transforms_data == True:
            data_transform = self.transform(True, True, True)
            image = data_transform(image)

        return image, torch.from_numpy(np.array(label_cross, dtype=np.long))#, image_path

    def transform(self, resize, totensor, normalize):
        options = []

        # if flip:
        #     print('oh nooooooooooooo')
        #     options.append(transforms.RandomHorizontalFlip())
        if resize:
            options.append(transforms.Resize(self.image_size))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        transform = transforms.Compose(options)

        return transform
