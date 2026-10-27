import sys
import time

sys.path.insert(0, "pytorch-generative")

import torch
from torch.nn import functional as F

from mnist_data import get_mnist_loaders
from pytorch_generative import models

train, _ = get_mnist_loaders(256, dequantize=True)
x, y = next(iter(train))
model = models.NICE(
    n_features=784, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
)
print("NICE params", sum(p.numel() for p in model.parameters()), "x", tuple(x.shape))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
preds_z, log_det = model(x)
log_prob = -(F.softplus(preds_z) + F.softplus(-preds_z)).sum(dim=(1, 2, 3))
loss = -(log_prob + log_det).mean()
loss.backward()
opt.step()
print("NICE fwd+bwd", round(time.time() - t0, 3), "s  loss", float(loss))

train, _ = get_mnist_loaders(256, dynamically_binarize=True, resize_to_32=True)
x, y = next(iter(train))
vae = models.VAE(
    in_channels=1,
    out_channels=1,
    latent_channels=16,
    strides=[2, 2, 2, 2],
    hidden_channels=64,
    residual_channels=32,
)
print("VAE params", sum(p.numel() for p in vae.parameters()), "x", tuple(x.shape))
opt = torch.optim.Adam(vae.parameters(), lr=5e-4)
t0 = time.time()
preds, kl = vae(x)
recon = F.binary_cross_entropy_with_logits(preds, x, reduction="none").sum(dim=(1, 2, 3))
loss = (recon + kl).mean()
loss.backward()
opt.step()
print("VAE fwd+bwd", round(time.time() - t0, 3), "s  loss", float(loss))
