"""
Problem: Generative modeling with an independence assumption (binarized MNIST).

Model: every pixel x_i in {0,1} is treated as an independent Bernoulli(theta_i)
random variable, so the whole image factorizes as
    p(x; theta) = prod_{i=1}^{D} theta_i^{x_i} (1 - theta_i)^{1 - x_i},   D = 784.

We derive the MLE for theta from a training set of N binarized images, fit it
on (a binarized version of) MNIST, and use the fitted model to generate new
samples by drawing each pixel independently from Bernoulli(theta_i_hat).
"""

import gzip
import os
import pickle
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Load MNIST and binarize it with a deterministic threshold at 0.5
# (pixel -> 1 ("on"/black) if intensity > 0.5). Note this is a deliberate
# simplification, not the fixed *stochastic* Larochelle binarization that
# the standard "binarized_mnist" dataset (e.g. via TFDS) ships (each pixel
# sampled once as Bernoulli(intensity)); the deterministic threshold is
# used here instead for simplicity and full reproducibility.
# ----------------------------------------------------------------------

MNIST_URL = (
    "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
    "master/data/mnist.pkl.gz"
)
if not os.path.exists("mnist.pkl.gz"):
    print("Downloading MNIST (~17MB)...")
    urllib.request.urlretrieve(MNIST_URL, "mnist.pkl.gz")

with gzip.open("mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

X_train, y_train = train_set  # X_train: (50000, 784) floats in [0, 1]
N, D = X_train.shape
H = W = 28
print(f"Loaded MNIST training set: N={N} images, D={D} pixels ({H}x{W}).")

X_bin = (X_train > 0.5).astype(np.float64)  # binarize -> {0, 1}
print(f"Fraction of 'on' pixels overall: {X_bin.mean():.4f}")

# ----------------------------------------------------------------------
# Maximum likelihood estimate of theta (per pixel Bernoulli parameter).
#
# Derivation (see accompanying writeup for full steps):
#   log L(theta) = sum_n sum_i [ x_i^(n) log theta_i + (1 - x_i^(n)) log(1 - theta_i) ]
#   dlogL/dtheta_i = sum_n [ x_i^(n)/theta_i - (1 - x_i^(n))/(1 - theta_i) ] = 0
#   => theta_i_hat = (1/N) sum_n x_i^(n)   (the empirical per pixel mean)
# ----------------------------------------------------------------------

theta_hat = X_bin.mean(axis=0)  # shape (784,), the MLE
print(f"theta_hat range: [{theta_hat.min():.4f}, {theta_hat.max():.4f}]")

# Small numerical safeguard purely for later log-likelihood evaluation
# (pixels that are always 0 or always 1 in the training set would otherwise
# produce -inf log likelihood on unseen data with the opposite value).
eps = 1e-6
theta_safe = np.clip(theta_hat, eps, 1 - eps)

# Report the average training log likelihood per image under the fitted model.
log_lik_per_image = (
    X_bin @ np.log(theta_safe) + (1 - X_bin) @ np.log(1 - theta_safe)
)
print(f"Average training log likelihood per image: {log_lik_per_image.mean():.2f} nats")

# ----------------------------------------------------------------------
# Visualize the learned parameter vector theta_hat as a 28x28 "mean image".
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3.2, 3.2))
im = ax.imshow(theta_hat.reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
ax.set_title(r"Learned $\hat\theta$ (per pixel P(black))")
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("theta_hat.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# Generate 10 new images: draw every pixel independently from Bernoulli(theta_hat_i).
# ----------------------------------------------------------------------

n_samples = 10
samples = rng.binomial(1, theta_hat, size=(n_samples, D)).astype(np.float64)

fig, axes = plt.subplots(1, n_samples, figsize=(1.7 * n_samples, 2.0))
for j, ax in enumerate(axes):
    ax.imshow(samples[j].reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("10 samples from the pixel independence model")
fig.tight_layout()
fig.savefig("independence_samples.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# For comparison: a few real (binarized) MNIST training images.
# ----------------------------------------------------------------------

real_idx = rng.choice(N, size=10, replace=False)
fig, axes = plt.subplots(1, 10, figsize=(1.7 * 10, 2.0))
for j, ax in enumerate(axes):
    ax.imshow(X_bin[real_idx[j]].reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("10 real binarized MNIST digits (for comparison)")
fig.tight_layout()
fig.savefig("real_examples.png", dpi=150)
plt.close(fig)

print("\nSaved: theta_hat.png, independence_samples.png, real_examples.png")
