"""Requirement 4 (part 1): single neurons.

McCulloch-Pitts neuron (1943): binary inputs, fixed weights, hard threshold. Nothing is learned.
A grid search shows that NO single threshold neuron computes XOR (XOR is not linearly separable).
Sigmoid neuron: a smooth threshold that CAN be trained by gradient descent (here it learns OR).
"""
import numpy as np

from fundamentals.common import lines, out_path


class McCullochPitts:
    """y = 1 if sum_i w_i x_i >= theta else 0. The designer chooses w and theta."""

    def __init__(self, w, theta):
        self.w, self.theta = np.asarray(w, dtype=float), theta

    def fire(self, *x):
        return int(np.dot(self.w, x) >= self.theta)


def AND():
    return McCullochPitts([1, 1], 2)


def OR():
    return McCullochPitts([1, 1], 1)


def NOT():
    return McCullochPitts([-1], 0)


def search_xor_solutions():
    """Try every (w1, w2, theta) on a grid from -3 to 3 in steps of 0.5.
    Returns (number of settings tried, number that compute XOR). Expected: 0 solutions."""
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Y = [0, 1, 1, 0]
    grid = np.arange(-3, 3.001, 0.5)
    tried = found = 0
    for w1 in grid:
        for w2 in grid:
            for th in grid:
                tried += 1
                n = McCullochPitts([w1, w2], th)
                if all(n.fire(*x) == y for x, y in zip(X, Y)):
                    found += 1
    return tried, found


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class SigmoidNeuron:
    """y_hat = sigmoid(w.x + b), trained with binary cross-entropy (BCE)
    L = -[y log y_hat + (1 - y) log(1 - y_hat)].
    Gradient: dL/dz = y_hat - y, dL/dw = (y_hat - y) x, dL/db = y_hat - y."""

    def __init__(self, n_in):
        self.w = np.zeros(n_in)
        self.b = 0.0

    def forward(self, X):
        return sigmoid(X @ self.w + self.b)

    def train_epoch(self, X, Y, lr):
        """One full-batch gradient-descent step. Returns the mean BCE before the update."""
        p = self.forward(X)
        loss = float(np.mean(-(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12))))
        dz = p - Y
        self.w -= lr * (X.T @ dz) / len(X)
        self.b -= lr * dz.mean()
        return loss


def run(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    summary = {}
    say("\n================ NEURONS: McCulloch-Pitts and sigmoid neuron ================")
    a, o, n = AND(), OR(), NOT()
    say("x1 x2 | AND OR")
    table = []
    for x1 in (0, 1):
        for x2 in (0, 1):
            table.append((x1, x2, a.fire(x1, x2), o.fire(x1, x2)))
            say(f" {x1}  {x2} |  {a.fire(x1, x2)}   {o.fire(x1, x2)}")
    say(f"NOT(0) = {n.fire(0)}  NOT(1) = {n.fire(1)}")
    summary["AND_OR_truth_table"] = table
    summary["NOT"] = [n.fire(0), n.fire(1)]
    tried, found = search_xor_solutions()
    summary["xor_grid_settings_tried"] = tried
    summary["xor_grid_solutions_found"] = found
    say(f"Grid search for one MP neuron computing XOR: {tried} settings tried, {found} solutions")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([0, 1, 1, 1], dtype=float)                 # OR
    sn = SigmoidNeuron(2)
    loss = [sn.train_epoch(X, Y, lr=1.0) for _ in range(500)]
    outputs = sn.forward(X)
    summary["sigmoid_OR_loss_first"] = loss[0]
    summary["sigmoid_OR_loss_last"] = loss[-1]
    summary["sigmoid_OR_outputs"] = [round(float(p), 3) for p in outputs]
    say(f"Sigmoid neuron learning OR: loss {loss[0]:.4f} -> {loss[-1]:.4f}; outputs "
        + " ".join(f"{p:.3f}" for p in outputs))
    lines(out_path("neurons", "sigmoid_or_loss.png"), "Sigmoid neuron on OR (BCE loss)", "epoch", "loss",
          ["train loss"], [loss])
    return summary


if __name__ == "__main__":
    run()
