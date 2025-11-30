import torch
import torch.nn as nn
from model.EncoderClass import EncoderClass as VAEencoder
from model.DecoderClass import DecoderClass as VAEdecoder

class VAEclass(nn.Module):

    def __init__(self, latent_dim=32, beta = 1.0):
        super(VAEclass, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = VAEencoder(latent_dim)
        self.decoder = VAEdecoder(latent_dim)

    # Gradients flow freely from loss → z → mu and σ → encoder parameters
    # eps is not dependent on mean or variance which earlier required sampling of z
    def reparameterisation(self, mean, log_var, is_training=True):
        # get standard deviation (σ) from log of variance
        std = torch.exp(0.5*log_var)

        # create random sampling via external gaussian distribution as eps
        eps = torch.randn_like(std)  # same size of std

        if is_training:
            reparam = mean + eps * std
            return reparam
        else:
            # for test-time, we do not need randomness
            return mean

    def forward(self, x):

        mean, log_var = self.encoder(x)
        # random latent variable
        z = self.reparameterisation(mean, log_var)

        recons = self.decoder(z)

        return recons, z, mean, log_var

    def sample(self, num_samples, device: torch.device):
        # Now, using the prior distribution of latent variable z, p(z) = N(0,I)
        # we will generate the new images
        z = nn.randn(num_samples, self.latent_dim, device=device)

        # generate new samples at test-time
        with torch.no_grad():
            samples = self.decoder(z)

        return samples

    def reconstruct_eval(self, x):
        # set in evaluation mode for metric evaluation of reconstruction
        self.eval()
        with torch.no_grad():
            mean, log_var = self.encoder(x)
            # re-parameterization without randomness for reconstruction
            z = self.reparameterisation(mean, log_var, is_training=False)
            recons = self.decoder(z)

        return recons

