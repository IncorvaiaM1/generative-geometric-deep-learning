"""
Problem 1: Data generation in low and high dimensions.

Part (a): exact generative model for 3x3 binary images, p(x) proportional to
          number of black pixels.
Part (b): learn the same distribution from 200 (image, true probability) pairs
          using an MLP, then build a categorical distribution from its scores.
Part (c): discussion is in the accompanying writeup (problem1_writeup.md);
          this script only demonstrates the numbers referenced there.
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Part (a): exact distribution over all 3x3 binary images
# ----------------------------------------------------------------------

N = 3  # image side length
D = N * N  # number of pixels = 9

# Enumerate all 2^9 = 512 possible binary images. Convention: 1 = black pixel.
all_images = np.array(list(itertools.product([0, 1], repeat=D)), dtype=np.float32)
num_images = all_images.shape[0]
print(f"Total number of possible {N}x{N} binary images: {num_images}")

# Unnormalized weight = number of black pixels in the image.
black_counts = all_images.sum(axis=1)  # shape (512,)

# Normalize to a valid pmf.
Z = black_counts.sum()
p_true = black_counts / Z
print(f"Normalizing constant Z = sum of black pixel counts over all images = {Z}")
print(f"Sanity check, probabilities sum to: {p_true.sum():.6f}")
print(f"Min/max probability: {p_true.min():.6f} / {p_true.max():.6f}")

# Draw 5 samples using a Categorical distribution over the 512 images.
sample_idx_a = rng.choice(num_images, size=5, p=p_true, replace=True)
samples_a = all_images[sample_idx_a]


def plot_images(images, titles, fname, suptitle):
    fig, axes = plt.subplots(1, len(images), figsize=(2.2 * len(images), 2.6))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.reshape(N, N), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


titles_a = [f"p={p_true[i]:.4f}\n({int(black_counts[i])} black)" for i in sample_idx_a]
plot_images(samples_a, titles_a, "part_a_samples.png",
            "Part (a): 5 samples from the exact distribution")

# ----------------------------------------------------------------------
# Part (b): learn the distribution from 200 (image, true probability) pairs
# ----------------------------------------------------------------------

# Sample 200 UNIQUE images according to p_true (weighted sampling w/o replacement).
# Only 511 images have nonzero probability (the all white image has p=0),
# so 200 unique draws is well within range.
train_idx = rng.choice(num_images, size=200, replace=False, p=p_true)
X_train = all_images[train_idx]
y_true_prob = p_true[train_idx]

# Train the MLP to predict log p(x) (probabilities span several orders of
# magnitude, so regressing in log space is much better conditioned than
# regressing the raw probabilities). The score output can be turned back into
# a valid distribution over all images with a softmax, which is exactly the
# hint about using a Categorical distribution.
eps = 1e-12
y_train = np.log(y_true_prob + eps)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=5000,
    n_iter_no_change=200,
    tol=1e-8,
    random_state=0,
)
mlp.fit(X_train, y_train)
train_pred = mlp.predict(X_train)
print(f"MLP fit on training set: corr={np.corrcoef(train_pred, y_train)[0, 1]:.4f}, "
      f"iters={mlp.n_iter_}, final loss={mlp.loss_:.2e}")

# Evaluate the trained MLP on ALL possible 3x3 images (512 of them).
scores_all = mlp.predict(all_images)  # predicted log probability like score

# Normalize with a softmax over the full 512-image space -> valid pmf.
scores_shift = scores_all - scores_all.max()
p_learned = np.exp(scores_shift)
p_learned = p_learned / p_learned.sum()

print(f"\nLearned distribution sums to: {p_learned.sum():.6f}")

# Draw 5 samples from the learned distribution.
sample_idx_b = rng.choice(num_images, size=5, p=p_learned, replace=True)
samples_b = all_images[sample_idx_b]
titles_b = [f"p_hat={p_learned[i]:.4f}\n({int(black_counts[i])} black)" for i in sample_idx_b]
plot_images(samples_b, titles_b, "part_b_samples.png",
            "Part (b): 5 samples from the MLP learned distribution")

# A quick fidelity check: how well does the learned distribution track the
# true one, in terms of ranking by number of black pixels?
corr = np.corrcoef(p_true, p_learned)[0, 1]
print(f"Correlation between p_true and p_learned across all 512 images: {corr:.4f}")

# Save a comparison plot of true vs. learned probability vs. black pixel count.
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(black_counts, p_true, alpha=0.4, label="true p(x)", s=15)
ax.scatter(black_counts, p_learned, alpha=0.4, label="MLP learned p_hat(x)", s=15)
ax.set_xlabel("number of black pixels")
ax.set_ylabel("probability")
ax.legend()
ax.set_title("True vs. MLP learned distribution (3x3 images)")
fig.tight_layout()
fig.savefig("part_b_comparison.png", dpi=150)
plt.close(fig)

print("\nSaved: part_a_samples.png, part_b_samples.png, part_b_comparison.png")
