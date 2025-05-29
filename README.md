# Variational Autoencoder (VAE) — Mathematical and PyTorch Implementation

This repository contains an educational and rigorous implementation of a **Variational Autoencoder (VAE)** using PyTorch, grounded in its mathematical formulation and applied to the MNIST dataset.

---

## 🧠 What is a VAE?

A variational autoencoder (VAE) is a latent-variable statistical model with observable variables \( \mathbf{x} \) and latents \( \mathbf{z} \), where the marginal likelihood is typically written as:

\[
p_{\\theta}(\\mathbf{x}) = \\int p_{\\theta}(\\mathbf{x} \\mid \\mathbf{z}) p(\\mathbf{z}) \\, d\\mathbf{z}
\]

The integral is generally intractable, since the generator \( p_{\\theta}(\\mathbf{x} \\mid \\mathbf{z}) \) is parametrised by a neural network, called the **decoder**, with parameters \( \\theta \). Moreover, the classical EM algorithm is also intractable since neither the posterior distribution \( p_{\\theta}(\\mathbf{z} \\mid \\mathbf{x}) \) nor the computation of the lower-bound in the E-step is tractable.

Instead, VAEs use **variational inference** to optimise the statistical model \( p_\\theta \). We define \( q_{\\phi}(\\mathbf{z} \\mid \\mathbf{x}) \) to be the variational approximation of \( p_{\\theta}(\\mathbf{z} \\mid \\mathbf{x}) \), and write the variational lower-bound for a single data point \( \\mathbf{x}^i \):

\[
\\mathcal{L}_i(\\theta, \\phi) = \\mathbb{E}_{q_{\\phi}(\\mathbf{z} \\mid \\mathbf{x}^i)}\\left[\\log \\frac{p_{\\theta}(\\mathbf{x}^i \\mid \\mathbf{z}) p(\\mathbf{z})}{q_{\\phi}(\\mathbf{z} \\mid \\mathbf{x}^i)}\\right]
\]

The lower-bound is then maximised with respect to the parameters \( \\theta \) and \( \\phi \) iteratively until convergence using stochastic gradient ascent. VAEs parameterise the variational distribution \( q_{\\phi}(\\mathbf{z} \\mid \\mathbf{x}) \) using a neural network — the **encoder** — with globally shared parameters \( \\phi \).

---

## 🎯 Project Goals

- Implement the mathematical equations in code
- Implement ways of fitting variational autoencoders
- Fit and experiment with VAEs on MNIST data

---

## 📈 VAE for MNIST

A large portion of MNIST pixel values are either 0 or 1, but the number of values between 0 and 1 is non-zero. Therefore, the support of the generative model should be \([0, 1]\). It is a [common error](https://proceedings.neurips.cc/paper/2019/hash/f82798ec8909d23e55679ee26bb26437-Abstract.html) in the VAE literature to use the Bernoulli distribution (support: {0,1}) for such data.

Instead, we parameterise the generator with a **continuous Bernoulli distribution**, whose support lies in \([0, 1]\):

\[
p_{\\theta}(\\mathbf{x} \\mid \\mathbf{z}) = \\mathcal{CB}(\\mathbf{x}; \\eta) = \\prod_{d=1}^D \\mathcal{CB}(x_d; \\eta_d), \\quad \\eta_d = \\text{sigmoid}(f_{\\theta}(\\mathbf{z})_d)
\]

Here, \( \\eta_d \\in (0, 1) \) are the parameters of the distribution. The neural network \( f_\\theta(\cdot) \) takes \( \\mathbf{z} \) as input and outputs the logits. The sigmoid function maps these logits to the interval \( (0, 1) \).

---

## 📦 Latent Variable Modeling

We choose the prior distribution over the latents as a **standard Gaussian**:

\[
p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{z}; \\mathbf{0}, \\mathbf{I})
\]

The variational distribution family is a **diagonal Gaussian**:

\[
q_{\\phi}(\\mathbf{z} \\mid \\mathbf{x}) = \\mathcal{N}(\\mathbf{z}; \\mu, \\text{diag}(\\sigma^2)) = \\prod_{k=1}^H \\mathcal{N}(z_k; \\mu_k, \\sigma_k)
\]

where:

\[
\\sigma_k = \\exp(0.5 \\gamma_{\\phi}(\\mathbf{x})_k), \\quad \\mu_k = \\mu_{\\phi}(\\mathbf{x})_k
\]

\( \\mu \\) and \( \\sigma \) are the outputs of the encoder network, which parameterizes the mean and standard deviation of the variational distribution.

---

## ✏️ Gradients and the Reparameterization Trick

To optimise the statistical model \( p_\\theta \), we aim to maximise the lower-bound with respect to \( \\theta \) and \( \\phi \). The gradient with respect to \( \\theta \) is tractable since the generator is parametrised by a neural network.

However, the first expectation \( \\mathbb{E}_{q_\\phi}[\\cdot] \) depends on the parameters \( \\phi \), which complicates the computation of its gradients.

We implement the most common way for obtaining gradients of the variational model: the **pathwise gradient estimator**, which reparametrises the latent random variable \( \\mathbf{z} \\sim q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \) in terms of a noise variable \( \\boldsymbol{\\epsilon} \\sim p(\\boldsymbol{\\epsilon}) \), and writes

\[
\\mathbf{z} = g_\\phi(\\boldsymbol{\\epsilon}, \\mathbf{x})
\]

where \( g_\\phi \) is a deterministic function parametrised by \( \\phi \).

This allows us to evaluate gradients of expectations by reparameterising the expectation as:

\[
\\mathbb{E}_{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}[f(\\mathbf{z})] = \\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[f(g_\\phi(\\boldsymbol{\\epsilon}, \\mathbf{x}))]
\]

So that we can take gradients inside the expectation:

\[
\\nabla_\\phi \\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[f(g_\\phi(\\boldsymbol{\\epsilon}, \\mathbf{x}))]
\]

---

## 📁 Notebook Contents

- `VAE.ipynb`: Full notebook including:
  - Theoretical derivation of VAEs
  - Description of variational inference and ELBO
  - Explanation and implementation of the reparameterization trick
  - Encoder/decoder architecture in PyTorch
  - Training on MNIST and visualizations of latent space

---

