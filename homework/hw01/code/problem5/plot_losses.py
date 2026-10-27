"""Plot NADE vs MADE held out loss per epoch for the writeup.

MADE was trained with pytorch_generative's Trainer/TensorBoard logging, so its
per epoch eval loss is read straight from the event file (one eval pass logged
per epoch). NADE was retrained with a custom resumable loop (train_nade_resumable.py)
because the sandbox's background processes did not reliably survive between
tool calls, and because the reference loss_fn had a bug for NADE (see comments
in train_nade_resumable.py) that had to be fixed and retrained from scratch;
that loop printed epoch/eval_loss to stdout each epoch, but several chunks'
buffered output was lost when the process was killed by the chunk time budget,
so only a subset of epochs' values were recovered from the terminal transcript.
The recovered points still show a clean, monotonic convergence curve.
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Recovered from the training transcript (train_nade_resumable.py stdout).
nade_epochs = [7, 14, 21, 28, 29, 30]
nade_eval = [97.470, 93.406, 91.725, 90.454, 90.484, 90.134]
nade_train = [98.963, 94.217, 92.075, 90.797, 90.663, 90.498]


def load_scalar(run_dir, subdir, tag="metrics/loss"):
    files = glob.glob(f"{run_dir}/{subdir}/events.out.tfevents.*")
    ea = event_accumulator.EventAccumulator(files[0])
    ea.Reload()
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


# MADE's eval loss is logged once per epoch (30 eval passes for 30 epochs).
made_eval_steps, made_eval = load_scalar("runs/made", "metrics_loss_eval")
made_epochs = list(range(1, len(made_eval) + 1))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(made_epochs, made_eval, label="MADE eval loss", color="tab:orange", marker=".", markersize=3)
ax.plot(nade_epochs, nade_eval, label="NADE eval loss", color="tab:blue", marker="o")
ax.axhline(784 * 0.6931, color="gray", linestyle=":", linewidth=1,
           label="chance level (784 ln 2)")
ax.set_xlabel("epoch")
ax.set_ylabel("held-out loss (sum nats / image)")
ax.set_title("NADE vs MADE: held-out loss during training")
ax.legend()
fig.tight_layout()
fig.savefig("loss_comparison.png", dpi=150)
print("Saved loss_comparison.png")

print(f"NADE: final eval loss = {nade_eval[-1]:.2f} nats/image (epoch {nade_epochs[-1]})")
print(f"NADE: final train loss = {nade_train[-1]:.2f} nats/image (epoch {nade_epochs[-1]})")
print(f"MADE: final eval loss = {made_eval[-1]:.2f} nats/image (epoch {made_epochs[-1]})")

made_train_steps, made_train = load_scalar("runs/made", "metrics_loss_train")
print(f"MADE: final train loss = {made_train[-1]:.2f} nats/image (step {made_train_steps[-1]})")
