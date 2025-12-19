Vector Quantisation - Variational AutoEncoder
This is an extension of the standard VAE which can learn discrete latent representation through vector 
quantisation. By quantizing the latent space into a finite set of discrete codes (like a codebook),
VQ-VAE enables more compact, interpretable, and efficient representations, making it suitable for multi-modalities.

The standard VAE has below issues:
1. Posterior Collapse : Instability of model learning where decoder becomes very powerful that the model ignores z,
setting the variance to zero and solely relying on the decoder's learning.
2. Blurry images: Using Gaussian Prior assumptions makes the generated image blurry due to smoothness
property of the gaussian function.

Further, as more data such as text, speech, and image patches, are discrete in nature, it is important to 
learn discrete representation.
1. It enables compression and compactness
2. It supports autoregressive priors for high quality generation
3. Avoid collapse by forcing the model to use a finite codebook.

