import os
import sys
import json
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from model.VAEclass import VAEclass
from torchvision import datasets
from argparse import ArgumentParser, Namespace
from utils.utils import imageTransformPipeline


# Give any two input image, we use these images as base to interpolate their features
# to generate new samples randomly
# Using decoder, we will generate the image out of randomly sampled noise
# Sample z~N(0,I) -> Decoder = generated image
def generate_interpolated_image(device, config, model_chkpt_path, num_samples):
    # Get test images
    vae_transform = imageTransformPipeline()
    test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=vae_transform)

    dataset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # define model
    latent_dim = config["latent_dim"]
    model = VAEclass(latent_dim).to(device)

    checkpoint = torch.load(model_chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # evaluation mode

    print("checkpoint loaded properly")

    with torch.no_grad():
        next_batch,_ = next(iter(dataset_loader))
        image_batch = next_batch[:2].to(device)

        # get mean and variance for latent space distribution for both images
        mean, log_var = model.encoder(image_batch)

        # get latent vector conditioned on both images => mean (in case of is_Training=False)
        z1 = model.reparameterisation(mean[0], log_var[0], is_training=False)
        z2 = model.reparameterisation(mean[1], log_var[1], is_training=False)

        @torch.no_grad()
        def interpolate(model, z1, z2, steps=4):
            model.eval()
            # Linear Interpolation: get steps values between 0 and 1  = [0.0, 0.25, 0.50, 0.75, 1.0]
            alphas = torch.linspace(0, 1, steps).to(z1.device)
            images = []

            for alpha in alphas:
                z = z1 * alpha + (1-alpha) * z2
                samples = model.decoder(z.unsqueeze(0))
                images.append(samples)

            # N interpolations combination using torch.cat
            # (1, C, H, W) * N -> (N, C, H, W)
            return torch.cat(images)

        img_samples = interpolate(model, z1, z2, steps=4)



    print("Interpolated Samples generated properly")
    # visualize the generated image
    grid = vutils.make_grid(img_samples.cpu(), nrow=4, normalize=True, padding=2)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(f"VAE Generated Fashion-MNIST interpolated samples (latent_dim={latent_dim})")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.tight_layout()
    # plt.show()

    folder_name = os.path.basename(os.path.dirname(model_chkpt_path))
    output_path = "./outputs/interpolate_generated_images/" + folder_name
    os.makedirs(output_path, exist_ok=True)
    vutils.save_image(img_samples.cpu(), f"{output_path}/vae_interpolated_generated_samples.png", nrow=2, normalize=True)
    print("Saved: vae_generated_interpolated_samples.png")


if __name__ == "__main__":

    # training command: python3 generate_interpolated_image.py --config_file scripts/config.json --num_samples 4 --saved_model_path checkpoints/vae_exp1/vae_chkpt_e100.pth
    print("------------ Variational Autoencoder training started -----------")
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_file", type=str, default="scripts/config.json", help="Path to the configuration file")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--saved_model_path", type=str, default="./checkpoints/vae_exp1/vae_chkpt_e100.pth",
                        help="Path to saved checkpoint")
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

    model_path = args.saved_model_path
    num_samples = args.num_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generate_interpolated_image(device, config, model_path, num_samples)