"""Load a trained model's weights and generate/visualize samples."""
import argparse
import sys

sys.path.insert(0, "pytorch-generative")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pytorch_generative import models

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["nade", "made"], required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--n_samples", type=int, default=16)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.model == "nade":
    model = models.NADE(input_dim=784, hidden_dim=args.hidden_dim)
else:
    model = models.MADE(input_dim=784, hidden_dims=[args.hidden_dim], n_masks=1)

state = torch.load(args.weights, map_location="cpu")
model.load_state_dict(state)
model.eval()

samples = model.sample(n_samples=args.n_samples)
print("samples shape:", samples.shape, "min/max:", samples.min().item(), samples.max().item())

n = args.n_samples
cols = min(n, 10)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
axes = axes.flatten() if n > 1 else [axes]
for i in range(n):
    axes[i].imshow(samples[i, 0].numpy(), cmap="gray_r", vmin=0, vmax=1)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
for i in range(n, len(axes)):
    axes[i].axis("off")
fig.suptitle(f"{args.model.upper()} samples (hidden_dim={args.hidden_dim}, 30 epochs)")
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"Saved to {args.out}")
