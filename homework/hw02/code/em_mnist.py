"""EM for a mixture of independent Bernoullis on binarized MNIST."""

from __future__ import annotations

import gzip
import os
import pickle
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

MNIST_URL = (
    "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
    "master/data/mnist.pkl.gz"
)

ALPHA1 = 1e-8
ALPHA2 = 1e-8
LOG_CLIP = 1e-12


def load_binarized_mnist(n: int = 1000, seed: int = 0, data_path: str = "mnist.pkl.gz"):
    """Return N random binarized MNIST digits as an (N, 784) array in {0, 1}."""
    if not os.path.exists(data_path):
        print(f"Downloading MNIST to {data_path} ...")
        urllib.request.urlretrieve(MNIST_URL, data_path)

    with gzip.open(data_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    x_train = np.concatenate([train_set[0], valid_set[0]], axis=0)
    y_train = np.concatenate([train_set[1], valid_set[1]], axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.choice(x_train.shape[0], size=n, replace=False)
    xs = (x_train[idx] > 0.5).astype(np.float64)
    ys = y_train[idx]
    return xs, ys


def Estep(Xs, p, mix_pi):
    """E-step: posterior cluster responsibilities eta[i, k] = eta(z_k^{(i)}).

    Computed in log space:
        log P(x^{(i)}, z_k=1) = log pi_k + sum_d [ x_d log p_{kd} + (1-x_d) log(1-p_{kd}) ]
    then softmax over k.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), LOG_CLIP, 1.0 - LOG_CLIP)
    mix_pi = np.clip(np.asarray(mix_pi, dtype=np.float64).reshape(-1), LOG_CLIP, 1.0)

    log_px = Xs @ np.log(p).T + (1.0 - Xs) @ np.log(1.0 - p).T
    log_joint = np.log(mix_pi)[None, :] + log_px

    log_joint -= log_joint.max(axis=1, keepdims=True)
    unnormalized = np.exp(log_joint)
    eta = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    return eta


def Mstep(Xs, eta, alpha1, alpha2):
    """M-step with Dirichlet / Beta-style smoothing to avoid log(0)."""
    Xs = np.asarray(Xs, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    n, d = Xs.shape
    k = eta.shape[1]

    n_k = eta.sum(axis=0)
    p = (eta.T @ Xs + alpha1) / (n_k[:, None] + alpha1 * d)
    mix_pi = (n_k + alpha2) / (n_k.sum() + alpha2 * k)
    return p, mix_pi.reshape(-1, 1)


def EM(Xs, K, iter=20, seed=0, alpha1=ALPHA1, alpha2=ALPHA2):
    """Run EM for `iter` iterations. p is initialized Unif(0,1) then L2-normalized."""
    xs = np.asarray(Xs, dtype=np.float64)
    n, d = xs.shape
    rng = np.random.default_rng(seed)

    p = rng.uniform(0.0, 1.0, size=(K, d))
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    mix_pi = np.full((K, 1), 1.0 / K)

    for _ in range(iter):
        eta = Estep(xs, p, mix_pi)
        p, mix_pi = Mstep(xs, eta, alpha1, alpha2)
    return p, mix_pi


def plot_bernoulli_params(p, title, path, mix_pi=None):
    """Grid of mean images, at most 5 per row; label each panel with k (and pi_k)."""
    k, d = p.shape
    side = int(np.sqrt(d))
    cols = min(5, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    pis = None if mix_pi is None else np.asarray(mix_pi).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < k:
            ax.imshow(p[i].reshape(side, side), cmap="gray", vmin=0.0, vmax=1.0)
            label = f"k={i}" if pis is None else f"k={i}, \u03c0={pis[i]:.3f}"
            ax.set_title(label, fontsize=13)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_images(p, mix_pi, n_samples=20, seed=1):
    """Ancestral sampling: k ~ pi, then x_d ~ Bernoulli(p_{kd}) independently."""
    mix_pi = np.asarray(mix_pi).reshape(-1)
    mix_pi = mix_pi / mix_pi.sum()
    rng = np.random.default_rng(seed)
    ks = rng.choice(len(mix_pi), size=n_samples, p=mix_pi)
    samples = rng.binomial(1, p[ks])
    return samples, ks


def plot_samples(samples, title, path):
    n, d = samples.shape
    side = int(np.sqrt(d))
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(samples[i].reshape(side, side), cmap="gray", vmin=0.0, vmax=1.0)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    xs, ys = load_binarized_mnist(n=1000, seed=0)
    print("Xs shape:", xs.shape, "mean pixel on:", xs.mean())

    for k in (5, 10, 20):
        print(f"Running EM with K={k} ...")
        p, mix_pi = EM(xs, K=k, iter=20, seed=0)
        mix_pi_flat = mix_pi.reshape(-1)
        print(f"  pi: {np.array2string(mix_pi_flat, precision=3)}")
        plot_bernoulli_params(
            p,
            title=f"Bernoulli parameters p  (K={k})",
            path=f"results/bernoulli_params_K{k}.png",
            mix_pi=mix_pi,
        )
        samples, _ = sample_images(p, mix_pi, n_samples=20, seed=1)
        plot_samples(
            samples,
            title=f"Samples from mixture of Bernoullis  (K={k})",
            path=f"results/samples_K{k}.png",
        )
        np.savez(f"results/em_K{k}.npz", p=p, mix_pi=mix_pi, ys=ys)

    print("Saved plots to results/")


if __name__ == "__main__":
    main()
