"""Requirement 4 (part 2): a multilayer perceptron with manual backpropagation.

Shapes (N = batch size):
  X  (N, d0)
  Z_l = A_{l-1} W_l + b_l       W_l (d_{l-1}, d_l), b_l (1, d_l)  <- b is broadcast over the N rows
  A_l = act(Z_l)                hidden: tanh / sigmoid / relu, output: sigmoid
  Loss = mean BCE + (lambda / 2) * sum ||W_l||^2

Backpropagation (chain rule, from the last layer backwards):
  dZ_L = (A_L - Y) / N                     (sigmoid + BCE simplify to this)
  dW_l = A_{l-1}^T dZ_l + lambda W_l
  db_l = sum over rows of dZ_l
  dA_{l-1} = dZ_l W_l^T
  dZ_{l-1} = dA_{l-1} * act'(Z_{l-1})       (* = element-wise)

Each training run saves its hyperparameters (hparams.json) and per-epoch metrics (metrics.csv).
"""
import csv
import json

import numpy as np

from fundamentals.common import lines, out_path, scatter, xor_data
from fundamentals.neurons import sigmoid
from fundamentals.tensor_sandbox import max_relative_error, numerical_grad

ACTIVATIONS = {
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
}
INITS = ["zero", "small_normal", "xavier", "he"]


class MLP:
    def __init__(self, sizes, act="tanh", init="xavier", seed=0, l2=0.0):
        self.sizes, self.act_name, self.init, self.seed, self.l2 = list(sizes), act, init, seed, l2
        self.act, self.act_grad = ACTIVATIONS[act]
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            std = {"zero": 0.0,
                   "small_normal": 0.01,
                   "xavier": np.sqrt(2.0 / (fan_in + fan_out)),   # keeps variance stable for tanh/sigmoid
                   "he": np.sqrt(2.0 / fan_in)}[init]             # keeps variance stable for ReLU
            self.W.append(rng.standard_normal((fan_in, fan_out)) * std)
            self.b.append(np.zeros((1, fan_out)))
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        """Returns predicted probabilities (N, 1) and keeps Z and A of every layer for backward()."""
        self.Zs, self.As = [], [X]
        A = X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b                                    # broadcasting adds the bias row to every sample
            A = sigmoid(Z) if l == len(self.W) - 1 else self.act(Z)
            self.Zs.append(Z)
            self.As.append(A)
        return A

    def loss(self, P, Y):
        """Mean binary cross-entropy + L2 penalty."""
        P = np.clip(P, 1e-12, 1 - 1e-12)
        bce = float(np.mean(-(Y * np.log(P) + (1 - Y) * np.log(1 - P))))
        return bce + 0.5 * self.l2 * sum(float((w * w).sum()) for w in self.W)

    def backward(self, Y):
        """Backpropagation. Call right after forward(X). Fills self.dW and self.db (in place)."""
        L = len(self.W)
        dZ = (self.As[L] - Y) / len(Y)
        for l in range(L - 1, -1, -1):
            self.dW[l][...] = self.As[l].T @ dZ + self.l2 * self.W[l]
            self.db[l][...] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dA = dZ @ self.W[l].T
                dZ = dA * self.act_grad(self.Zs[l - 1])

    def params(self):
        return self.W + self.b

    def grads(self):
        return self.dW + self.db

    def predict_proba(self, X):
        return self.forward(X).ravel()

    def accuracy(self, X, Y):
        return float(np.mean((self.forward(X) >= 0.5) == (Y >= 0.5)))


class Optimizer:
    """theta = parameters, g = gradient, lr = learning rate, mu = momentum.
      gd        theta <- theta - lr g          (Batch GD and SGD differ only in the batch size)
      momentum  v <- mu v + g ; theta <- theta - lr v
      nesterov  v <- mu v + g ; theta <- theta - lr (g + mu v)   ("look ahead", PyTorch form)"""

    def __init__(self, kind="gd", lr=0.1, mu=0.0):
        self.kind, self.lr, self.mu = kind, lr, mu
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            if self.kind == "gd":
                p -= self.lr * g
            elif self.kind == "momentum":
                v[...] = self.mu * v + g
                p -= self.lr * v
            elif self.kind == "nesterov":
                v[...] = self.mu * v + g
                p -= self.lr * (g + self.mu * v)
            else:
                raise ValueError(self.kind)

    def describe(self):
        return {"optimizer": self.kind, "lr": self.lr, "momentum": self.mu}


def train(net, opt, train_data, val_data=None, epochs=100, batch_size=32, seed=0, run_dir=None):
    """Mini-batch training loop. batch_size = N -> Batch GD; batch_size = 1 -> SGD.
    The data are reshuffled every epoch with a fixed seed. Returns a dict with the curves."""
    X, Y = train_data
    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch_size):
            idx = order[s:s + batch_size]
            net.forward(X[idx])
            net.backward(Y[idx])
            opt.step(net.params(), net.grads())
        hist["train_loss"].append(net.loss(net.forward(X), Y))
        hist["train_acc"].append(net.accuracy(X, Y))
        if val_data is not None:
            hist["val_loss"].append(net.loss(net.forward(val_data[0]), val_data[1]))
            hist["val_acc"].append(net.accuracy(*val_data))
    hist["final_train_acc"] = hist["train_acc"][-1]
    hist["final_val_acc"] = hist["val_acc"][-1] if val_data is not None else None
    if run_dir is not None:
        save_run(net, opt, epochs, batch_size, seed, hist, run_dir)
    return hist


def save_run(net, opt, epochs, batch_size, seed, hist, run_dir):
    """Experiment logging: hyperparameters -> hparams.json, per-epoch metrics -> metrics.csv."""
    folder = out_path("runs", run_dir, "x").parent
    hparams = {"layers": net.sizes, "hidden_activation": net.act_name, "init": net.init, "init_seed": net.seed,
               **opt.describe(), "epochs": epochs, "batch_size": batch_size, "l2": net.l2, "shuffle_seed": seed,
               "final_train_acc": hist["final_train_acc"], "final_val_acc": hist["final_val_acc"]}
    (folder / "hparams.json").write_text(json.dumps(hparams, indent=2))
    with open(folder / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for e in range(epochs):
            w.writerow([e, f"{hist['train_loss'][e]:.6f}", f"{hist['train_acc'][e]:.4f}",
                        f"{hist['val_loss'][e]:.6f}" if hist["val_loss"] else "",
                        f"{hist['val_acc'][e]:.4f}" if hist["val_acc"] else ""])


def gradient_check(net, X, Y, h=1e-5):
    """Compare backprop gradients with central-difference gradients for every parameter."""
    net.forward(X)
    net.backward(Y)
    analytic = [g.copy() for g in net.grads()]
    worst = 0.0
    for p, g in zip(net.params(), analytic):
        num = numerical_grad(p, lambda: net.loss(net.forward(X), Y), h)
        worst = max(worst, max_relative_error(num, g))
    return worst


def run(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    summary = {}
    say("\n================ MLP: forward, backprop, gradient check, XOR ================")
    X, Y = xor_data()
    net = MLP([2, 4, 1], act="tanh", init="xavier", seed=7)
    summary["backprop_gradcheck_max_rel_error"] = gradient_check(net, X, Y)
    net_l2 = MLP([2, 4, 1], act="sigmoid", init="xavier", seed=7, l2=0.01)
    summary["backprop_gradcheck_max_rel_error_sigmoid_l2"] = gradient_check(net_l2, X, Y)
    ok = max(summary["backprop_gradcheck_max_rel_error"], summary["backprop_gradcheck_max_rel_error_sigmoid_l2"]) < 1e-6
    say(f"Backprop vs numerical gradient: max relative error = {summary['backprop_gradcheck_max_rel_error']:.2e} "
        f"(tanh), {summary['backprop_gradcheck_max_rel_error_sigmoid_l2']:.2e} (sigmoid + L2)  "
        f"({'PASS' if ok else 'FAIL'})")

    hist = train(net, Optimizer("gd", lr=0.5), (X, Y), epochs=2000, batch_size=4, seed=1, run_dir="mlp/xor")
    P = net.predict_proba(X)
    summary["xor_accuracy"] = hist["final_train_acc"]
    summary["xor_outputs"] = [round(float(p), 3) for p in P]
    summary["xor_final_loss"] = hist["train_loss"][-1]
    say(f"XOR after 2000 epochs: accuracy = {hist['final_train_acc']:.2f}; outputs "
        + " ".join(f"{p:.3f}" for p in P))
    lines(out_path("mlp", "xor_loss.png"), "MLP 2-4-1 on XOR", "epoch", "BCE", ["train"], [hist["train_loss"]], logy=True)
    scatter(out_path("mlp", "xor_boundary.png"), "XOR decision regions (MLP 2-4-1)", X, Y.ravel(), net.predict_proba)
    return summary


if __name__ == "__main__":
    run()
