import torch
import torch.nn as nn
from torchvision import transforms

def imageTransformPipeline():
    vae_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    return vae_transforms