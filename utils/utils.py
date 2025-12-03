import torch
import torch.nn as nn
from torchvision import transforms

def imageTransformPipeline(is_train = False):

    if is_train:
        vae_transforms = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
    else:
        vae_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    return vae_transforms