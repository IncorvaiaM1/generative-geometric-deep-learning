"""MNIST loaders matching pytorch-generative's interface, using the local pickle."""

import gzip
import os
import pickle
import urllib.request

import torch
from torch import distributions
from torch.nn import functional as F
from torch.utils import data

MNIST_URL = (
    "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
    "master/data/mnist.pkl.gz"
)


def _ensure_mnist(path):
    if not os.path.exists(path):
        print(f"Downloading MNIST to {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)


def _dequantize(x):
    return (x * 255 + torch.rand(x.shape)) / 256


def _resize_to_32(x):
    return F.pad(x, (2, 2, 2, 2))


class MNISTTensor(data.Dataset):
    def __init__(self, x, y, dynamically_binarize=False, dequantize=False, resize_to_32=False):
        self.x = torch.from_numpy(x).float().view(-1, 1, 28, 28)
        self.y = torch.from_numpy(y).long()
        self.dynamically_binarize = dynamically_binarize
        self.dequantize = dequantize
        self.resize_to_32 = resize_to_32

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.dynamically_binarize:
            x = distributions.Bernoulli(probs=x).sample()
        if self.dequantize:
            x = _dequantize(x)
        if self.resize_to_32:
            x = _resize_to_32(x)
        return x, self.y[idx]


def get_mnist_loaders(
    batch_size,
    dynamically_binarize=False,
    dequantize=False,
    resize_to_32=False,
    data_path="mnist.pkl.gz",
    num_workers=0,
):
    if dynamically_binarize and dequantize:
        raise ValueError("Cannot specify both dynamically_binarize and dequantize.")

    _ensure_mnist(data_path)
    with gzip.open(data_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    x_train, y_train = train_set
    x_test, y_test = test_set

    kwargs = dict(
        dynamically_binarize=dynamically_binarize,
        dequantize=dequantize,
        resize_to_32=resize_to_32,
    )
    train_ds = MNISTTensor(x_train, y_train, **kwargs)
    test_ds = MNISTTensor(x_test, y_test, **kwargs)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
