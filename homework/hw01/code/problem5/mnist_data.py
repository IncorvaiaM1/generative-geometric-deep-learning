"""By using this cusotom MNIST loader that avoids torchvision's network download (blocked in this
sandbox) by reusing the pickled MNIST file we already fetched from GitHub. Produces
DataLoaders with the exact same (x, y) image-batch interface torchvision's MNIST
dataset provides, so it's a drop in replacement for
`pytorch_generative.datasets.get_mnist_loaders`.
"""

import gzip
import os
import pickle
import urllib.request

import numpy as np
import torch
from torch import distributions
from torch.utils import data

MNIST_URL = (
    "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
    "master/data/mnist.pkl.gz"
)


def _ensure_mnist(path):
    if not os.path.exists(path):
        print(f"Downloading MNIST to {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)


class MNISTTensor(data.Dataset):
    def __init__(self, x, y, dynamically_binarize=False):
        # x: (N, 784) float32 in [0, 1]. Reshape to image format (N, 1, 28, 28).
        self.x = torch.from_numpy(x).float().view(-1, 1, 28, 28)
        self.y = torch.from_numpy(y).long()
        self.dynamically_binarize = dynamically_binarize

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.dynamically_binarize:
            x = distributions.Bernoulli(probs=x).sample()
        return x, self.y[idx]


def get_mnist_loaders(batch_size, dynamically_binarize=True, data_path="mnist.pkl.gz",
                       num_workers=0):
    _ensure_mnist(data_path)
    with gzip.open(data_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    x_train, y_train = train_set
    x_test, y_test = test_set

    train_ds = MNISTTensor(x_train, y_train, dynamically_binarize=dynamically_binarize)
    test_ds = MNISTTensor(x_test, y_test, dynamically_binarize=dynamically_binarize)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders(64)
    x, y = next(iter(train_loader))
    print("batch x shape:", x.shape, "dtype:", x.dtype, "min/max:", x.min().item(), x.max().item())
    print("batch y shape:", y.shape)
    print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
