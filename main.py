import os
import torch
import torch.nn.functional as F
from dataset.DatasetLoader import DatasetLoader
from utils.utils import imageTransformPipeline
from model.VAEclass import VAEclass
import wandb


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

def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    recons_losses = 0
    kl_losses = 0

    beta = 1.0 #get_beta(epoch)

    for batch_idx, (img_data, _) in enumerate(train_loader):
        data = img_data.to(device)  # [B, 1, 28, 28]

        # initialize optimizer
        optimizer.zero_grad()
        recon_batch, z, mu, logvar = model(data)

        loss_dict = VAE_Loss(data, recon_batch, mu, logvar, beta=beta)
        loss = loss_dict['vae_loss']
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        recons_losses += loss_dict['recons_loss'].item()
        kl_losses += loss_dict['kl_loss'].item()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | Recons: {loss_dict['recons_loss'].item()/len(data):.4f} | "
                  f"KL: {loss_dict['kl_loss'].item()/len(data):.4f} | β: {beta:.3f}")

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = recons_losses / len(train_loader.dataset)
    avg_kl = kl_losses / len(train_loader.dataset)

    return {
        "train/loss": avg_loss,
        "train/recons_loss": avg_recon,
        "train/kl_loss": avg_kl,
        "train/beta": beta
    }


# FashionMNIST contains 70000 grayscale images of size 28x28 of various classes of clothes
# Out of 70K, 60K images are for training and 10K images for testing.
def training_script(device, wandb, config):

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

    # define model
    latent_dim = 32
    model = VAEclass(latent_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        metrics = train_one_epoch(model, dataset_loader, optimizer, epoch)

        # Log metrics
        wandb.log(metrics, step=epoch)

        # Log images every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"Train Epoch: {epoch} : "
                  f"total Loss: {metrics['train/loss']:.4f} | Recons loss: {metrics['train/recons_loss']:.4f} | "
                  f"KL loss: {metrics['train/kl_loss']:.4f} ")

        # Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['train/loss'],
            }, f"vae_checkpoint_epoch{epoch}.pth")

    print("Training completed!")




if __name__ == "__main__":
    print("------------ Variational Autoencoder training started -----------")

    wandb.init(
        project="fashion-mnist-vae",
        config={
            "architecture": "VAE",
            "dataset": "FashionMNIST",
            "latent_dim": 32,
            "batch_size": 128,
            "epochs": 100,
            "learning_rate": 1e-3,
            "beta_start": 0.0,
            "beta_end": 1.0,
            "annealing_epochs": 30
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_script(device, wandb, config)