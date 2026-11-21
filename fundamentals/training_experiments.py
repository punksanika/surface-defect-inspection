"""Requirement 5: how training choices change the result (two-moons data).

1. Optimizers: Batch GD vs SGD vs Momentum vs Nesterov, all starting from the SAME weights.
2. Initialisation: zero vs small normal vs Xavier vs He in a deeper tanh network.
3. Generalisation: a big network on 30 points overfits; L2 regularisation and noise
   augmentation reduce the train/validation gap.
4. Input standardisation: features on wildly different scales, fixed with mean/std
   computed on the TRAINING set only (no leakage).
"""
import csv

import numpy as np

from fundamentals.common import augment, lines, moons, out_path, scatter, standardize, train_stats
from fundamentals.mlp import INITS, MLP, Optimizer, train


def optimizers(say):
    say("\n================ OPTIMIZERS: Batch GD vs SGD vs Momentum vs Nesterov (same init) ================")
    tr, va = moons(200, 0.15, 1), moons(200, 0.15, 2)
    configs = [("BatchGD", "gd", 0.5, 0.0, 200),
               ("SGD_b1", "gd", 0.05, 0.0, 1),
               ("Momentum", "momentum", 0.5, 0.9, 200),
               ("Nesterov", "nesterov", 0.5, 0.9, 200)]
    rows, names, curves = [], [], []
    for name, kind, lr, mu, batch in configs:
        net = MLP([2, 16, 1], act="tanh", init="xavier", seed=3)          # identical starting weights
        h = train(net, Optimizer(kind, lr, mu), tr, va, epochs=300, batch_size=batch, seed=5,
                  run_dir=f"optimizers/{name}")
        rows.append({"optimizer": name, "lr": lr, "momentum": mu, "batch": batch,
                     "final_train_loss": round(h["train_loss"][-1], 5), "val_acc": round(h["final_val_acc"], 4)})
        names.append(name)
        curves.append(h["train_loss"])
        say(f"{name:<9} lr={lr:<5} mu={mu:<4} batch={batch:<4} final train loss {h['train_loss'][-1]:.4f}  "
            f"val acc {h['final_val_acc']:.3f}")
    with open(out_path("optimizers", "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines(out_path("optimizers", "loss_curves.png"), "Optimizer comparison (train loss)", "epoch", "BCE",
          names, curves, logy=True)
    return rows


def initialisation(say):
    say("\n================ INITIALISATION: zero vs small normal vs Xavier vs He (deep tanh net) ================")
    tr, va = moons(200, 0.15, 1), moons(200, 0.15, 2)
    rows, names, curves = [], [], []
    for init in INITS:
        net = MLP([2, 32, 32, 32, 1], act="tanh", init=init, seed=3)
        h = train(net, Optimizer("momentum", 0.1, 0.9), tr, va, epochs=200, batch_size=32, seed=5,
                  run_dir=f"init/{init}")
        rows.append({"init": init, "final_train_loss": round(h["train_loss"][-1], 5),
                     "val_acc": round(h["final_val_acc"], 4)})
        names.append(init)
        curves.append(h["train_loss"])
        say(f"{init:<13} final train loss {h['train_loss'][-1]:.4f}  val acc {h['final_val_acc']:.3f}")
    lines(out_path("init", "loss_curves.png"), "Initialisation comparison", "epoch", "BCE", names, curves, logy=True)
    return rows


def generalisation(say):
    say("\n================ GENERALISATION: overfitting, L2, augmentation ================")
    small, va = moons(30, 0.25, 11), moons(400, 0.25, 12)
    rows, names, curves = [], [], []
    for setup in ["baseline", "L2=0.01", "augment x10"]:
        net = MLP([2, 64, 64, 1], act="tanh", init="xavier", seed=3, l2=0.01 if setup.startswith("L2") else 0.0)
        tr = augment(*small, k=10, sigma=0.1, seed=4) if setup.startswith("augment") else small
        h = train(net, Optimizer("momentum", 0.1, 0.9), tr, va, epochs=400, batch_size=16, seed=5,
                  run_dir="generalisation/" + setup.replace(" ", "_").replace("=", "_"))
        gap = h["final_train_acc"] - h["final_val_acc"]
        rows.append({"setup": setup, "train_acc": round(h["final_train_acc"], 4), "val_acc": round(h["final_val_acc"], 4),
                     "gap": round(gap, 4), "final_train_loss": round(h["train_loss"][-1], 5),
                     "final_val_loss": round(h["val_loss"][-1], 5)})
        say(f"{setup:<12} train acc {h['final_train_acc']:.3f}  val acc {h['final_val_acc']:.3f}  (gap {gap:.3f})  "
            f"train loss {h['train_loss'][-1]:.4f}  val loss {h['val_loss'][-1]:.4f}")
        names += [f"{setup} train", f"{setup} val"]
        curves += [h["train_loss"], h["val_loss"]]
        if setup == "baseline":
            scatter(out_path("generalisation", "overfit_boundary.png"), "Baseline: 30 points, big network",
                    small[0], small[1].ravel(), net.predict_proba)
    lines(out_path("generalisation", "train_vs_val.png"), "Overfitting: train vs validation loss", "epoch", "BCE",
          names, curves, logy=True)
    return rows


def normalisation(say):
    say("\n================ INPUT STANDARDISATION (train statistics only) ================")
    (Xtr, Ytr), (Xva, Yva) = moons(200, 0.15, 1), moons(200, 0.15, 2)
    scale, shift = np.array([[100, 0.01]]), np.array([[500, -3]])
    bad_tr, bad_va = Xtr * scale + shift, Xva * scale + shift            # features on very different scales
    stats = train_stats(bad_tr)                                          # from TRAIN only
    variants = {"unnormalised": ((bad_tr, Ytr), (bad_va, Yva)),
                "standardised": ((standardize(bad_tr, stats), Ytr), (standardize(bad_va, stats), Yva))}
    rows, names, curves = [], [], []
    for name, (tr, va) in variants.items():
        net = MLP([2, 16, 1], act="tanh", init="xavier", seed=3)
        h = train(net, Optimizer("momentum", 0.05, 0.9), tr, va, epochs=200, batch_size=32, seed=5,
                  run_dir=f"normalisation/{name}")
        rows.append({"input": name, "val_acc": round(h["final_val_acc"], 4),
                     "final_train_loss": round(h["train_loss"][-1], 5)})
        names.append(name)
        curves.append(h["train_loss"])
        say(f"{name:<13} val acc {h['final_val_acc']:.3f}  final train loss {h['train_loss'][-1]:.4f}")
    lines(out_path("generalisation", "normalisation.png"), "Input standardisation", "epoch", "BCE",
          names, curves, logy=True)
    return rows


def run(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    return {"optimizers": optimizers(say), "initialisation": initialisation(say),
            "generalisation": generalisation(say), "normalisation": normalisation(say)}


if __name__ == "__main__":
    run()
