import os
import sys
import json
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from model.VAEclass import VAEclass
from argparse import ArgumentParser, Namespace


# Using decoder, we will generate the image out of randomly sampled noise
# Sample z~N(0,I) -> Decoder = generated image
def generate_image(device, config, model_chkpt_path, num_samples):
    # define model
    latent_dim = config["latent_dim"]
    model = VAEclass(latent_dim).to(device)

    checkpoint = torch.load(model_chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # evaluation mode

    print("checkpoint loaded properly")

    with torch.no_grad():
        samples = model.generate_sample(num_samples, device)

    print("Samples generated properly")
    # visualize the generated image
    grid = vutils.make_grid(samples, nrow=2, normalize=True, padding=2)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(f"VAE Generated Fashion-MNIST Samples (latent_dim={latent_dim})")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.tight_layout()
    # plt.show()

    folder_name = os.path.basename(os.path.dirname(model_chkpt_path))
    output_path = "./outputs/generated_images/" + folder_name
    os.makedirs(output_path, exist_ok=True)
    vutils.save_image(samples, f"{output_path}/vae_generated_samples.png", nrow=2, normalize=True)
    print("Saved: vae_generated_samples.png")


if __name__ == "__main__":

    # training command: python3 test_generate_image.py --config_file scripts/config.json --num_samples 8 --saved_model_path checkpoints/vae_exp1/vae_chkpt_e100.pth
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

    generate_image(device, config, model_path, num_samples)