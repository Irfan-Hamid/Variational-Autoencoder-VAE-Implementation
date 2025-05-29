# Variational Autoencoder on MNIST

This project implements a Variational Autoencoder (VAE) in PyTorch to model the MNIST dataset by maximising the variational lower-bound using stochastic gradient ascent.

## Overview

- The variational distribution is defined as a diagonal Gaussian, parametrised by a neural network which outputs the mean and log-variance for each data point.
- The generator is defined using the continuous Bernoulli distribution, with parameters obtained by applying a sigmoid function to the output of a neural network.
- A standard Gaussian is used as the prior over the latent variables.
- The pathwise gradient estimator (reparametrisation trick) is used to express sampling from the variational distribution as a differentiable transformation of noise.

## Training

- The model is trained by maximising the variational lower-bound, also known as the ELBO.
- The KL divergence between the variational distribution and the prior is computed analytically.
- The model is trained by minimising the negative ELBO using the Adam optimiser.

## Evaluation

- Sampling is performed using ancestral sampling: first sample latent variables from the prior, then sample image data from the generator.
- The latent space is visualised by projecting data points using the mean of the variational distribution.
- Additional visualisations compare the structure of the latent space with generated samples.

## Project Goals

- Implement the mathematical equations of a VAE in code.
- Fit the model to MNIST using variational inference and gradient ascent.
- Visualise and evaluate the fitted model through sampling and latent projections.


