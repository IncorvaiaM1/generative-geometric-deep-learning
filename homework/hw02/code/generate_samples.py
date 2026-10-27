"""Generate sample grids from trained NICE and VAE checkpoints."""

import argparse
import sys

sys.path.insert(0, "pytorch-generative")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from pytorch_generative import models


def _grid(images, path, title, vmin=0.0, vmax=1.0):
    n = images.shape[0]
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.6 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0].cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def sample_nice(weights, n_samples, seed, out):
    torch.manual_seed(seed)
    model = models.NICE(
        n_features=784, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
    )
    state = torch.load(weights, map_location="cpu")
    # Trainer checkpoints wrap weights under "model"; plain state_dicts do not.
    if isinstance(state, dict) and "model" in state and "_c" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    # Shape buffers are required by sample(); NICE is trained on 1x28x28.
    if getattr(model, "_c", None) is None:
        model._create_shape_buffers(1, 28, 28)

    with torch.no_grad():
        samples = model.sample(n_samples=n_samples, temp=0.8)
    print("NICE samples min/max:", samples.min().item(), samples.max().item())
    samples = samples.clamp(0.0, 1.0)
    _grid(samples, out, "NICE samples (50 epochs, batch 256)")


def sample_vae(weights, n_samples, seed, out):
    torch.manual_seed(seed)
    model = models.VAE(
        in_channels=1,
        out_channels=1,
        latent_channels=16,
        strides=[2, 2, 2, 2],
        hidden_channels=64,
        residual_channels=32,
    )
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model" in state and "_c" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    if getattr(model, "_c", None) is None:
        model._create_shape_buffers(1, 32, 32)

    with torch.no_grad():
        logits = model._sample(n_samples)
        bernoulli = model.sample(n_samples)
        means = torch.sigmoid(logits)
    print("VAE logits min/max:", logits.min().item(), logits.max().item())
    _grid(bernoulli.float(), out, "VAE Bernoulli samples (50 epochs, batch 256)")
    mean_path = out.replace(".png", "_means.png")
    _grid(means, mean_path, "VAE decoder means (sigmoid of logits)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nice", "vae"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model == "nice":
        sample_nice(args.weights, args.n_samples, args.seed, args.out)
    else:
        sample_vae(args.weights, args.n_samples, args.seed, args.out)


if __name__ == "__main__":
    main()
