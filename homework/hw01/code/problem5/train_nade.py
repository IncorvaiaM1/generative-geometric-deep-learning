"""Train NADE on binarized MNIST for 30 epochs, hidden_dim=300, following the
repo's reproduce() function but swapping in our own MNIST loader (torchvision's
downloader is blocked in this sandbox) and hyperparameters per the assignment.
"""
import sys
import time

sys.path.insert(0, "pytorch-generative")

import torch
from torch import optim
from torch.nn import functional as F

from pytorch_generative import models, trainer
from mnist_data import get_mnist_loaders

BATCH_SIZE = 512
N_EPOCHS = 30
HIDDEN_DIM = 300
LOG_DIR = "runs/nade"

print(f"Building data loaders (batch_size={BATCH_SIZE})...")
train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, dynamically_binarize=True)
print(f"train batches/epoch={len(train_loader)}, test batches={len(test_loader)}")

model = models.NADE(input_dim=784, hidden_dim=HIDDEN_DIM)
n_params = sum(p.numel() for p in model.parameters())
print(f"NADE parameter count: {n_params:,}")

optimizer = optim.Adam(model.parameters())


def loss_fn(x, _, preds):
    # NADE's forward() returns p_hat (a probability, via an internal sigmoid),
    # not a logit -- see nade.py. The repo's reproduce() function pairs this
    # with binary_cross_entropy_with_logits, which double-applies a sigmoid
    # and stalls training near chance level. We use plain BCE instead, which
    # is the correct loss for NADE's probability output.
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    preds = preds.clamp(1e-7, 1 - 1e-7)
    loss = F.binary_cross_entropy(preds, x, reduction="none")
    return loss.sum(dim=1).mean()


model_trainer = trainer.Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    eval_loader=test_loader,
    log_dir=LOG_DIR,
    n_gpus=0,
    device_id=None,
)

print(f"Starting training for {N_EPOCHS} epochs...")
start = time.time()
model_trainer.interleaved_train_and_eval(N_EPOCHS)
elapsed = time.time() - start
print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

torch.save(model.state_dict(), "nade_final.pt")
print("Saved model weights to nade_final.pt")
