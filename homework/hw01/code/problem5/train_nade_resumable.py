"""Resumable NADE training loop that checkpoints after every epoch, so training
can proceed in short foreground chunks (the sandbox's background processes do
not reliably survive between tool calls / container recycles). Run this script
repeatedly; it picks up from the last saved checkpoint until N_EPOCHS is reached.
"""
import sys
import time
import argparse

sys.path.insert(0, "pytorch-generative")

import torch
from torch import optim
from torch.nn import functional as F

from pytorch_generative import models
from mnist_data import get_mnist_loaders

BATCH_SIZE = 512
N_EPOCHS = 30
HIDDEN_DIM = 300
CKPT_PATH = "nade_ckpt.pt"
FINAL_PATH = "nade_final.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--max_seconds", type=float, default=480.0,
                     help="Stop after roughly this many seconds of training in this invocation.")
args = parser.parse_args()

print(f"Building data loaders (batch_size={BATCH_SIZE})...")
train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, dynamically_binarize=True)
print(f"train batches/epoch={len(train_loader)}, test batches={len(test_loader)}")

model = models.NADE(input_dim=784, hidden_dim=HIDDEN_DIM)
n_params = sum(p.numel() for p in model.parameters())
print(f"NADE parameter count: {n_params:,}")

optimizer = optim.Adam(model.parameters())

start_epoch = 0
try:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]
    print(f"Resumed from checkpoint at epoch {start_epoch}")
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")


def loss_fn(x, preds):
    # NOTE: NADE's forward() already applies a sigmoid internally (it returns
    # p_hat, a probability, not a logit -- see nade.py's `_forward`). The
    # reference repo's reproduce() incorrectly feeds this through
    # binary_cross_entropy_with_logits, which silently re-applies a sigmoid to
    # an already-sigmoided value and squashes gradients so badly that training
    # stalls near chance level (~784*ln(2) nats). We use plain BCE on
    # probabilities here instead, which is the correct pairing for NADE's output.
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    eps = 1e-7
    preds = preds.clamp(eps, 1 - eps)
    loss = F.binary_cross_entropy(preds, x, reduction="none")
    return loss.sum(dim=1).mean()


def evaluate():
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in test_loader:
            preds = model(x)
            l = loss_fn(x, preds)
            total += l.item() * x.shape[0]
            n += x.shape[0]
    model.train()
    return total / n


start_time = time.time()
epoch = start_epoch
while epoch < N_EPOCHS:
    epoch_start = time.time()
    model.train()
    running = 0.0
    nb = 0
    for x, _ in train_loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(x, preds)
        loss.backward()
        optimizer.step()
        running += loss.item()
        nb += 1
    train_loss = running / nb
    eval_loss = evaluate()
    epoch += 1
    elapsed_epoch = time.time() - epoch_start
    print(f"epoch {epoch}/{N_EPOCHS}  train_loss={train_loss:.3f}  eval_loss={eval_loss:.3f}  ({elapsed_epoch:.1f}s)")

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss},
               CKPT_PATH)

    if time.time() - start_time > args.max_seconds:
        print(f"Hit time budget ({args.max_seconds}s) after epoch {epoch}; stopping this chunk.")
        break

if epoch >= N_EPOCHS:
    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Training complete after {epoch} epochs. Saved final weights to {FINAL_PATH}")
else:
    print(f"Chunk done. {epoch}/{N_EPOCHS} epochs complete so far. Re-run this script to continue.")
