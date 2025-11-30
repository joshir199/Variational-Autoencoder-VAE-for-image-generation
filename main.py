import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataset import DatasetLoader
from utils.utils import imageTransformPipeline


# calculate VAE ELBO loss which consist of reconstruction loss and closed form KL divergence loss
def VAE_Loss(input, recons, mean, log_var, beta=1.0):

    # get decoder reconstruction loss
    recons_loss = F.binary_cross_entropy(recons, input, reduction="sum")

    # get encoder KL-divergence loss
    kl_loss = 0.5 * torch.sum(mean.pow(2) + torch.exp(log_var) - 1 - log_var)

    # ELBO loss
    vae_loss = recons_loss - beta * kl_loss

    return {
        "vae_loss": vae_loss,
        "recons_loss": recons_loss,
        "kl_loss": kl_loss
    }


# FashionMNIST contains 70000 grayscale images of size 28x28 of various classes of clothes
# Out of 70K, 60K images are for training and 10K images for testing.
def training_script():

    annotation_path = os.path.join("dataset/fashionmnist/fashion-mnist_test.csv")
    image_path = os.path.join("dataset/fashionmnist/train-images-idx3-ubyte")
    vae_transform = imageTransformPipeline()
    train_dataset = DatasetLoader(
        annotations_file = annotation_path,
        image_dir = image_path,
        transforms = vae_transform
    )

    dataset_loader = torch.utils.data.Dataloader(
        train_dataset,
        batch_size=16,
        shuffle = True,
        num_workers = 4,
        pin_memory=True
    )




















if __name__ == "__main__":
    pprint("------------ Variational Autoencoder training started -----------")

    training_script()