import torch
import torch.nn as nn
import torch.nn.functional as F
from vq_vae.VQVAE_DecoderClass import VQVAE_DecoderClass as VQVAEdecoder
from vq_vae.VQVAE_EncoderClass import VQVAE_EncoderClass as VQVAEencoder

class VQ_VAE_Model(nn.Module):

    def __init__(
            self,
            latent_dim=32,  # D-dimensional vector
            num_embeddings=64,  # K number of vector
            beta = 1.0
    ):
        super(VQ_VAE_Model, self).__init__()

        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.encoder = VQVAEencoder(latent_dim)
        self.decoder = VQVAEdecoder(latent_dim)

        # Learnable codebook
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        # initialize the codebook using uniform distribution (prior)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)


    def quantise(self, z_e, is_training=True):

        if is_training:
            # batch x K
            distance = torch.sum(z_e**2, dim=1, keep_dim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_e, self.embedding.weight.t())
            # get the code with minimum distance
            encode_indices = torch.argmin(distance, dim=1)

            z_q = F.embedding(encode_indices, self.embedding.weight)  # [Batch, D]

            # loss functions
            # gradients only flow to e (the codebook), not back to the encoder.
            vq_loss = F.mse_loss(z_q, z_e.detach()) # update in codebook (||sg[z_e] - e||)


            commitment_loss = F.mse_loss(z_q.detach(), z_e) # update the encoders

            # Straight-through estimator
            # The encoder receives gradients as if quantization were identity
            z_q_st = z_e + (z_q - z_e).detach()  # during foward => z_q { .detach() do nothing } and during backward = dz_q_st/dz_e = 1 {.detach makes 2nd term=0}

            return z_q, z_q_st, vq_loss, commitment_loss, encode_indices


    def forward(self, x):

        # encode
        z_e = self.encoder(x)

        # quantize into codebook
        z_q, z_q_st, vq_loss, commitment_loss, encode_indices = self.quantise(z_e)

        # decode
        recons = self.decoder(z_q)

        return recons, z_q, vq_loss, commitment_loss, encode_indices

    @torch.no_grad()
    def encode_from_codebook(self, x):
        # getting the nearest codebook indices
        _, _, _, _, indices = self.quantise(self.encoder(x))

        return indices

    @torch.no_grad()
    def reconstruct_from_eval(self, indices):
        # set in evaluation mode for metric evaluation of reconstruction
        z_q = F.embedding(indices.flatten(), self.embedding.weight)

        recons = self.decoder(z_q)
        self.eval()
        with torch.no_grad():
            mean, log_var = self.encoder(x)
            # re-parameterization without randomness for reconstruction
            z = self.reparameterisation(mean, log_var, is_training=False)
            recons = self.decoder(z)

        return recons

