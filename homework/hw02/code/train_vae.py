"""Train VAE on dynamically binarized 32x32 MNIST: 50 epochs, batch size 256."""

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
LOG_DIR = "runs/vae"


def main():
    print(
        f"Building VAE data loaders (batch_size={BATCH_SIZE}, "
        "dynamically_binarize=True, resize_to_32=True)..."
    )
    train_loader, test_loader = get_mnist_loaders(
        BATCH_SIZE, dynamically_binarize=True, resize_to_32=True
    )
    print(f"train batches/epoch={len(train_loader)}, test batches={len(test_loader)}")

    model = models.VAE(
        in_channels=1,
        out_channels=1,
        latent_channels=16,
        strides=[2, 2, 2, 2],
        hidden_channels=64,
        residual_channels=32,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameter count: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    def loss_fn(x, _, preds):
        preds, kl_div = preds
        recon_loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        recon_loss = recon_loss.sum(dim=(1, 2, 3))
        elbo = recon_loss + kl_div
        return {
            "recon_loss": recon_loss.mean(),
            "kl_div": kl_div.mean(),
            "loss": elbo.mean(),
        }

    os.makedirs(LOG_DIR, exist_ok=True)
    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
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
            print(f"[VAE] epoch {model_trainer._epoch + 1}/{N_EPOCHS} eval {pretty}", flush=True)

    model_trainer._log_metrics = _log_metrics

    print(f"Starting VAE training for {N_EPOCHS} epochs...")
    start = time.time()
    model_trainer.interleaved_train_and_eval(N_EPOCHS)
    elapsed = time.time() - start
    print(f"VAE training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    torch.save(model.state_dict(), "vae_final.pt")
    print("Saved model weights to vae_final.pt")

    os.makedirs("results", exist_ok=True)
    from generate_samples import sample_vae

    sample_vae("vae_final.pt", n_samples=32, seed=0, out="results/vae_samples.png")


if __name__ == "__main__":
    main()
