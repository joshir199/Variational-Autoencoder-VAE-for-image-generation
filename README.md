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


