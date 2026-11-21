"""Shared helpers: where results go, the small toy datasets, and plotting.

Every dataset takes a seed, so every run gives exactly the same numbers.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "experiments" / "fundamentals"


def out_path(*parts):
    """Path inside experiments/fundamentals/, creating its folder if needed."""
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- toy datasets
def xor_data():
    """The four XOR points. Labels: 1 when exactly one input is 1."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, Y


def moons(n, noise, seed):
    """Two interleaving half-circles ("two moons"): not linearly separable, needs a hidden layer.
    Even rows = upper moon (label 0), odd rows = lower moon (label 1)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2))
    Y = np.zeros((n, 1))
    for i in range(n):
        t = np.pi * rng.random()
        upper = i % 2 == 0
        x0 = np.cos(t) if upper else 1 - np.cos(t)
        x1 = np.sin(t) if upper else 0.5 - np.sin(t)
        X[i, 0] = x0 + noise * rng.standard_normal()
        X[i, 1] = x1 + noise * rng.standard_normal()
        Y[i, 0] = 0 if upper else 1
    return X, Y


def train_stats(X):
    """Column mean and standard deviation. Call this on the TRAINING set only (no leakage)."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    return mean, std


def standardize(X, stats):
    """(x - mean) / std with the given (training) statistics."""
    mean, std = stats
    return (X - mean) / std


def augment(X, Y, k, sigma, seed):
    """Original data plus k noisy copies of every sample (Gaussian jitter); labels unchanged."""
    rng = np.random.default_rng(seed)
    xs, ys = [X], [Y]
    for _ in range(k):
        xs.append(X + sigma * rng.standard_normal(X.shape))
        ys.append(Y)
    return np.vstack(xs), np.vstack(ys)


# ---------------------------------------------------------------- plotting
def lines(path, title, xlabel, ylabel, names, curves, logy=False):
    """Several curves in one line plot."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, c in zip(names, curves):
        ax.plot(np.arange(len(c)), c, label=name, linewidth=1.6)
    if logy:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if len(names) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scatter(path, title, X, labels, prob_fn=None, protos=None):
    """2-D points coloured by label. prob_fn draws the model's decision regions behind them;
    protos marks prototypes (competitive learning) with black crosses."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if prob_fn is not None:
        pad = 0.5
        gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200),
                             np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200))
        P = prob_fn(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, P, levels=20, cmap="RdBu_r", alpha=0.35)
        ax.contour(gx, gy, P, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", s=22, edgecolors="k", linewidths=0.4)
    if protos is not None:
        ax.scatter(protos[:, 0], protos[:, 1], marker="X", s=220, c="black", label="prototypes")
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
