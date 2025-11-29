import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image

class DatasetLoader(Dataset):

    def __init__(self, annotations, image_dir, transforms=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations)
        self.img_dir = image_dir
        self.transforms = transforms
        self.target_transforms = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transforms:
            img = self.transforms(img)
        elif self.target_transforms:
            label = self.target_transforms(label)

        return img, label