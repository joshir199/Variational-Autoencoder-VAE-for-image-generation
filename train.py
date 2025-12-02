import os
import sys
import json
import torch
import torch.nn.functional as F
from utils.utils import imageTransformPipeline
from model.VAEclass import VAEclass
from torchvision import datasets
from argparse import ArgumentParser, Namespace

try:
    import wandb

    WANDB_FOUND = False
except ImportError:
    WANDB_FOUND = False


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return memory_used_values


def initialize_debugger(wandb_name=None):
    if WANDB_FOUND and wandb_name != None:
        wandb_project = wandb_name
        wandb_run_name = "fashion-mnist"
        id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
        # name = os.path.basename(args.model_path) if wandb_run_name is None else wandb_run_name
        name = os.path.basename(args.source_path) + '_' + str(id)
        wandb.init(
            project=wandb_project,
            dir=args.model_path,
            id=id
        )


# calculate VAE ELBO loss which consist of reconstruction loss and closed form KL divergence loss
# We maximize ELBO → we minimize the negative ELBO
# minimize -> D_kl - loglikelihood  => D_kl + BCE
def VAE_Loss(input, recons, mean, log_var, beta=1.0):
    # get decoder reconstruction loss
    # Negative Loglikelihood
    recons_loss = F.binary_cross_entropy(recons, input, reduction="sum")

    # get encoder KL-divergence loss
    kl_loss = 0.5 * torch.sum(mean.pow(2) + torch.exp(log_var) - 1 - log_var)

    # ELBO loss
    vae_loss = recons_loss + beta * kl_loss

    return {
        "vae_loss": vae_loss,
        "recons_loss": recons_loss,
        "kl_loss": kl_loss
    }


# Adding KL annealing to avoid the risk of posterior collapse (D_kl decrease very rapidly)
def get_beta(epoch, total_anneal_epochs=10):
    if epoch <= total_anneal_epochs:
        return min(1.0, epoch / total_anneal_epochs)
    return 1.0


def train_one_epoch(model, train_loader, optimizer, epoch, use_wandb):
    model.train()
    total_loss = 0
    recons_losses = 0
    kl_losses = 0

    beta = 1.0  # get_beta(epoch)

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

        # if batch_idx % 100 == 0:
        #    print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] "
        #          f"Loss: {loss.item():.4f} | Recons: {loss_dict['recons_loss'].item() / len(data):.4f} | "
        #          f"KL: {loss_dict['kl_loss'].item() / len(data):.4f} ")

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
def training_script(device, config, model_path, use_wandb):
    vae_transform = imageTransformPipeline()

    train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=vae_transform)

    dataset_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        # num_workers = 4,
        pin_memory=True
    )

    # define model
    latent_dim = config["latent_dim"]
    model = VAEclass(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["epochs"] + 1):
        metrics = train_one_epoch(model, dataset_loader, optimizer, epoch, use_wandb)

        # Log metrics
        if use_wandb:
            wandb.log(metrics, step=epoch)

        # print training details every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Train Epoch: {epoch} : "
                  f"total Loss: {metrics['train/loss']:.4f} | Recons loss: {metrics['train/recons_loss']:.4f} | "
                  f"KL loss: {metrics['train/kl_loss']:.4f} ")

        # Save checkpoint
        if epoch % 20 == 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['train/loss'],
            }, f"{model_path}/vae_checkpoint_epoch{epoch}.pth")

    print("Training completed!")


if __name__ == "__main__":

    # training command: python3 train.py --config_file scripts/config.json --wandb_name vae_exp2 --model_path checkpoints
    print("------------ Variational Autoencoder training started -----------")
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_file", type=str, default="scripts/config.json", help="Path to the configuration file")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="./checkpoints/exp_1", help="Path to store checkpoint")
    args = parser.parse_args(sys.argv[1:])

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    wandb_name = args.wandb_name
    model_path = args.model_path + '/' + wandb_name
    os.makedirs(model_path, exist_ok=True)

    if wandb_name is None:
        use_wandb = False
    else:
        use_wandb = True

    wandb_enabled = (WANDB_FOUND and use_wandb)

    initialize_debugger(wandb_name)

    if wandb_enabled:
        wandb.run.summary['GPU'] = torch.cuda.get_device_name(0).split()[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_script(device, config, model_path, wandb_enabled)