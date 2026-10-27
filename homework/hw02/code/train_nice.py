"""Train NICE on dequantized MNIST: 50 epochs, batch size 256, other params default."""

import os
import sys
import time

sys.path.insert(0, "pytorch-generative")

import torch
from torch import optim
from torch.nn import functional as F

from pytorch_generative import models, trainer
from mnist_data import get_mnist_loaders

N_EPOCHS = 50
BATCH_SIZE = 256
LOG_DIR = "runs/nice"


def main():
    print(f"Building NICE data loaders (batch_size={BATCH_SIZE}, dequantize=True)...")
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, dequantize=True)
    print(f"train batches/epoch={len(train_loader)}, test batches={len(test_loader)}")

    model = models.NICE(
        n_features=784, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NICE parameter count: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn(x, _, preds):
        preds, log_det_J = preds
        log_prob = -(F.softplus(preds) + F.softplus(-preds)).sum(dim=(1, 2, 3))
        loss = log_prob + log_det_J
        return {
            "loss": -loss.mean(),
            "prior_log_likelihood": log_prob.mean(),
            "log_det_J": log_det_J.mean(),
        }

    os.makedirs(LOG_DIR, exist_ok=True)
    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        lr_scheduler=None,
        log_dir=LOG_DIR,
        n_gpus=0,
        device_id=0,
        save_checkpoint_epochs=10,
    )

    orig_log = model_trainer._log_metrics

    def _log_metrics(metrics, training):
        orig_log(metrics, training)
        if not training:
            pretty = {k: f"{v:.4f}" for k, v in metrics.items()}
            print(f"[NICE] epoch {model_trainer._epoch + 1}/{N_EPOCHS} eval {pretty}", flush=True)

    model_trainer._log_metrics = _log_metrics

    print(f"Starting NICE training for {N_EPOCHS} epochs...")
    start = time.time()
    model_trainer.interleaved_train_and_eval(N_EPOCHS)
    elapsed = time.time() - start
    print(f"NICE training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    torch.save(model.state_dict(), "nice_final.pt")
    print("Saved model weights to nice_final.pt")

    os.makedirs("results", exist_ok=True)
    from generate_samples import sample_nice

    sample_nice("nice_final.pt", n_samples=32, seed=0, out="results/nice_samples.png")


if __name__ == "__main__":
    main()
