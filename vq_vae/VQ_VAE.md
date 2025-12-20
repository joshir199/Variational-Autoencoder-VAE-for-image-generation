# Vector Quantisation - Variational AutoEncoder
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


*********************************************************
# VQ-VAE Model Architecture

<div align="center">
    <img src="https://github.com/joshir199/Variational-Autoencoder-VAE-for-image-generation/blob/main/utils/images/vq_vae_architecture.png" alt="vq_vae architecture">
    <p><i>Vector Quantisation - Variational AutoEncoder (VQ-VAE) architecture diagram</i></p>
</div>

*********************************************************
# Mathematical Concepts & their formulation

1. **Encoder \(p(z<sub>e</sub>/x)\)** : It encodes the input (image/text) \(x\) into latent vector with dimensio equal to latent dimension \(D\). Let's say \(z<sub>e</sub>\) be the output vector from encoder E, then


     z<sub>e</sub> = E(x)  ; 
   
2. **Vector Quantisation** :  It quantizes z<sub>e</sub>(x) by mapping it to the nearest vector in a learned codebook **e** ∈ ℝ^{K × D}, where K is the codebook size.

     z<sub>q</sub> = e<sub>k</sub> ; where k = argmin<sub>j</sub>(|| z<sub>e</sub> - e<sub>j</sub> ||^2)
     discrete vector

3. **Decoder \(p(x/z<sub>q</sub>)\)** : Reconstructs original vector \(x\) from the quantized z<sub>q</sub>.
      
     x = D(z<sub>q</sub>)  ;

**Note**: Due to discretization of the latent vector, the gradient at quantisation part is **non-differentiable**, which blocks the gradient flow from decoder to encoder durig backpropagation. Thus, we need to pass the gradient using **Straight-Through-Estimator (STE)** which mean copying the gradients directly from decoder to encoder blocks while skipping the quantisation part.
*********************************************************************
# Training

The loss function in VQ-VAE is pivotal to its design, as it addresses the challenges of learning discrete representations while enabling end-to-end training of a model. The loss function exhibit following property:

1. Encourage accurate reconstruction of x from the quantized latents 

2. Update the codebook embeddings to better represent the data using its disentangled codebook latents

3. Force the encoder to produce outputs close to codebook entries, preventing divergence.

Also, Similar to standard VAE, its loss function includes reconstruction loss and KL divergence loss but VQ-VAE uses uniform distribution for latent prior, **the KL divergence term becomes constant** and thus can be ignored. 

**Full loss, L = log⁡ p(x∣z<sub>q</sub>(x)) + ∥sg[z<sub>e</sub>(x)] − e∥^2 + β∥z<sub>e</sub>(x) − sg[e]∥^2**;

where **sg[x] = stop-gradient operator** which treats x as constant, thus no gradient flow.

$$

During forward: 
sg[v] = v ;

During backward:
partial derivative (sg[v]) w.r.t v = 0;

$$

Thus, for 2nd VQ term: sg[z<sub>e</sub>] → 0  ==>  gradients only to e (codebook update)

And in 3rd term: sg[e] → 0  ==>  gradients only to z<sub>e</sub> (encoder update).



This loss is summed/averaged over the batch during training. The three different terms of loss function are:
1. **Reconstruction Loss (Data Term)** — log p(x|z<sub>q</sub>(x))
   The first term is the reconstruction loss which optimizes the decoder and the encoder (no learning for codebook latents). This term pushes the model (encoder, decoder) toward high-fidelity reconstructions from quantised latents. For images, it's per-pixel Bernoulli (for binary) and the loss is BCE.

2. **VQ Loss (Codebook Update)** - ∥sg[z<sub>e</sub>(x)] − e∥^2
    Due to the straight-through gradient estimation... the embeddings e_i receive no gradients from the reconstruction loss. Thus, this term learns the codebook by pulling embeddings e closer to encoder outputs z<sub>e</sub>(x), like online k-means clustering. During backward pass, it updates the parameters of latent codebook.

3. **Third Term: Commitment Loss** — β.∥z<sub>e</sub>(x) − sg[e]∥^2
    The embedding space has no fixed scale (dimensionless), so z<sub>e</sub>(x) could grow arbitrarily large if e updates slower than the encoder. This term penalizes the encoder for producing z<sub>e</sub> far from any codebook latent e, encouraging tight clustering around codes. During backward pass, it updates the parameters of encoder only.

### Gradient Flow: Who Updates What?
* Decoder: Only reconstruction term (gradients through x^).
* Encoder: Reconstruction (via STE) + commitment (pulls z<sub>e</sub> to e).
* latents (Codebook): Only VQ term (pulls e to z<sub>e</sub> clusters).












