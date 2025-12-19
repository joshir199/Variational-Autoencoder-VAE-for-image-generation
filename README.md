# Variational Autoencoder (VAE) for Image Generation
Understanding and building the Variational Autoencoder (VAE) for learning latent variable model and generating images. The project demonstrates the core concepts of VAEs, including encoding images into a latent space, reconstructing them, and generating new samples.

# VAE Introduction
A Generative framework that learns to represent data into lower dimensional latent space (latent probability distribution) which enables both reconstruction and generation of novel samples. Unlike traditional autoencoder which learns only single latent vector, VAE learns the probability distibution of latent space which generates the relevant latent vectors. Mathematically, It learns to approximate the posterior distribution over latent variables.

The true aim of VAE is to learn best representations in latent space often called as Disentangled representation learning:
* Semantically meaningful : Changing a latent dimension does something which can be descibed in words (e.g: "make the color red", "rotate the hand to left", etc.)
* Disentangled: Each latent dimension should control one meaningful feature, and only that feature.
* Statistically independent: Changing one latent feature should not accidentally change others.
* Causal: The dimensions of latent vector correspond to actual causes in the real world.  
*********************************************************
# VAE Model Architecture

<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/utils/images/vae_architecture.png" alt="vae architecture">
    <p><i>Variational AutoEncoder (VAE) architecture diagram</i></p>
</div>

*********************************************************
# Mathematical Concepts & their formulation
### Latent Variable Model (LVMs)

VAEs are based on latent variable models where observed data \( x \) is generated from hidden (latent) variables \( z \). The joint distribution is:

$$
p_\theta(x, z) = p_\theta(x|z) \. p(z)
$$

with prior p(z) = N(0, I).
The marginal likelihood is:

$$
p_\theta(x) = \int p_\theta(x|z) \. p(z) \ dz
$$

This integral is **intractable**, so we introduce an approximate posterior q(z|x, ϕ) to infer latent variable z.

### 1. Inference Part (Encoder)
The encoder approximates:

$$
q_\phi(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)I)
$$

It outputs:
- Mean: μ
- Log-variance: log(σ^2) (choosen because of numerically stable)

This neural network compresses the input image into probabilistic latent parameters. We are using single encoder to perform posterior inference over all inputs. Thus, the learned weights of this encoder is shored for each input data, often the process called amortized variational inference.  

### Generative Part (Decoder)
The decoder models the conditional likelihood which reconstructs the original image given latent variable z:

$$
p_\theta(x|z)
$$

Typically a **Bernoulli** distribution (for binary-like pixels in [0,1]) or Gaussian.  
It uses transposed convolutions to reconstruct the image from latent variable z, ending with a **sigmoid** activation to output pixel probabilities.

#### Reparameterization Trick
To enable backpropagation through stochastic sampling (earlier not possible due to gradient w.r.t to ϕ cannot enter inside the expectation over q(z/ϕ) to enable Monte-carlo sampling):

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

This moves randomness outside the computation graph — gradients flow through μ and σ, but z remains stochastic.

#### Loss Function and ELBO
The VAE optimizes the **Evidence Lower BOund (ELBO)**:

$$
\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)]}_{\text{Reconstruction term}} - \beta \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL regularization}}
$$

- Reconstruction term is equivalent to binary cross-entropy (BCE) for pixel likelihood

- KL Divergence (analytical closed form for diagonal Gaussians) is given below:
  
$$
D_{KL} = \frac{1}{2} \sum_{j=1}^D \left( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \right)
$$


And, the final training loss is:

$$
\text{Loss} = \text{Reconstruction Loss} + \beta \cdot \text{KL Divergence}
$$

where β controls the trade-off (used in **β-VAE** for disentangled representations).

### Applications of VAE:
* **Image Generation**: Sampling from latent space for novel images.
* **Data Compression**: Latent representations for efficient storage.
* **Data Augmentation**: Create new variants of existing dataset for improving training dataset diversity.
* **Disentangled Representations**: Learning unique Factors like style, pose in images.
* **Semi-Supervised Learning**: Use latent space for downstream tasks.

*******************************************************************
# Training

* Dataset details: We used opensource dataset Fashion-MNIST, which consist of 28x28 grayscale image dataset with 60,000 training and 10,000 test samples across 10 classes (e.g., t-shirts, trousers, bags).
* Training details: The model is trained on GPU/CPU with Adam optimizer (lr=1e-3), batch size 128. KL annealing ramped β from 0 to 1 over 30 epochs to prevent posterior collapse.

  <div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/training/vae_exp1_train_loss.png" alt="vae  exp1 train loss">
    <p><i>Figure 1: Train loss per epoch for vae experiment_1 with fixed beta = 1.</i></p>
</div>

<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/training/vae_exp2_train_loss.png" alt="vae  exp2 train loss">
    <p><i>Figure 2: Train loss per epoch for vae experiment_2 with KL annealing (till 30 epoch, later beta =1).</i></p>
</div>
    
<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/training/KL_divergence_loss_for_different_beta.png" alt="vae  exp3 KL divergence loss">
    <p><i>Figure 3: KL-Divergence loss per epoch for vae experiment_3 as compared with experiment_2 with KL annealing (till 30 epoch). The loss depicts the behaviour of KL-divergence loss term during better disentanglement of latent variables.</i></p>
</div>

# Results
- **Image Generation from Noise**
  Samples generated by sampling z ~ N(0,I) and decoding. Diverse items like shoes, dresses, shirts appear realistic.
  
<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/generated_images/vae_exp1/vae_generated_samples.png" alt="vae  exp1 generated image">
    <p><i>Figure 4: Generated images from vae exp_1.</i></p>
</div>
    
<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/generated_images/vae_exp2/vae_generated_samples.png" alt="vae  exp2 generated image">
    <p><i>Figure 5: Generated images from vae exp_2.</i></p>
</div>

- **Image Interpolation**
 Linear interpolation in latent space between two encoded images shows smooth transitions, demonstrating structured latent space.
 
<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/interpolate_generated_images/vae_exp1/vae_latent_interpolation_with_originals.png" alt="vae  exp1 interpolated image">
    <p><i>Figure 6: Generated images from vae exp_1 (leftmost: image A, rightmost: image B and between: generated images after interpolation).</i></p>
</div>

    
<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/interpolate_generated_images/vae_exp2/vae_latent_interpolation_with_originals.png" alt="vae  exp2 interpolated image">
    <p><i>Figure 7: Generated images from vae exp_2 (leftmost: image A, rightmost: image B and between: generated images after interpolation).</i></p>
</div>
    
*******************************************************
# Disentanglement Analysis of VAE latent variable 

* β=1: Standard VAE (balances reconstruction and regularization).
* β>1: Stronger penalty on KL → forces even more independence and alignment to the prior. The model is heavily pressured to make q(z|x) factorize independently (diagonal covariance) and match the prior closely.
* β<1: Weaker KL → better reconstruction but more entangled latents (useful for high-fidelity generation but less interpretability).

To evaluate disentanglement, traverse each latent dimension while fixing others at mean and vary selected one from -3σ to +3σ. Further, visualize if it controls one factor cleanly.



<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/generated_images/vae_exp3_beta4/vae_generated_samples.png" alt="vae  exp3 interpolated image">
    <p><i>Figure 8: Generated images from vae exp_3 (with beta = 4). As compared to exp2, the generated images are poorer when disentanglement is increased because reconstruction loss is given less importance.</i></p>
</div>

<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/outputs/interpolate_generated_images/vae_exp3_beta4/vae_latent_interpolation_with_originals.png" alt="vae  exp3 interpolated image">
    <p><i>Figure 9: Generated images from vae exp_3 (with beta = 4) (leftmost: image A, rightmost: image B and between: generated images after interpolation). As compared to exp2, the generated images from interpolation are better and diverse when disentanglement is increased because KL-divergence loss is given more importance.</i></p>
</div>












