"""Requirement 6: classical learning rules and an energy-based model.

y = w . x is a single linear neuron unless stated otherwise.
  Hebbian           dw = eta y x               "fire together, wire together"; ||w|| grows without limit
  Oja               dw = eta y (x - y w)       Hebb + decay; ||w|| -> 1 and w -> FIRST PRINCIPAL COMPONENT
  Competitive       winner k = argmin_j ||x - w_j||, dw_k = eta (x - w_k)    (only the winner learns)
  Perceptron/delta  dw = eta (t - y) x         learns from the ERROR (the root of backpropagation)
Hopfield network: stores patterns; a corrupted pattern rolls "downhill" in energy to the stored one.
"""
import matplotlib.pyplot as plt
import numpy as np

from fundamentals.common import lines, out_path, scatter
from fundamentals.tensor_sandbox import svd_jacobi


# ================================================================ data
def correlated_data(n, deg, seed):
    """2-D Gaussian cloud stretched along angle `deg` (std 3 along it, 0.5 across), centred at 0."""
    rng = np.random.default_rng(seed)
    a = np.radians(deg)
    u, v = 3.0 * rng.standard_normal(n), 0.5 * rng.standard_normal(n)
    return np.c_[np.cos(a) * u - np.sin(a) * v, np.sin(a) * u + np.cos(a) * v]


def blobs(n_per, centers, std, seed):
    """n_per points around each centre; returns points and their cluster labels."""
    rng = np.random.default_rng(seed)
    X = np.vstack([np.asarray(c) + std * rng.standard_normal((n_per, 2)) for c in centers])
    labels = np.repeat(np.arange(len(centers)), n_per)
    return X, labels


# ================================================================ rules
def hebbian(X, w, eta, epochs):
    """Returns ||w|| after every epoch; w is updated in place."""
    norms = []
    for _ in range(epochs):
        for x in X:
            y = w @ x
            w += eta * y * x
        norms.append(float(np.linalg.norm(w)))
    return norms


def oja(X, w, eta, epochs):
    norms = []
    for _ in range(epochs):
        for x in X:
            y = w @ x
            w += eta * y * (x - y * w)
        norms.append(float(np.linalg.norm(w)))
    return norms


def pca_direction(X):
    """First principal direction from OUR OWN Jacobi SVD (rows = samples, data centred first)."""
    _, _, V = svd_jacobi(X - X.mean(axis=0))
    return V[:, 0]


def abs_cosine(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def competitive(X, k, eta, epochs, seed):
    """Winner-take-all. Prototypes start at random samples (avoids 'dead' units that never win)."""
    rng = np.random.default_rng(seed)
    w = X[rng.integers(len(X), size=k)].copy()
    for _ in range(epochs):
        for _ in range(len(X)):
            x = X[rng.integers(len(X))]
            win = int(np.argmin(((w - x) ** 2).sum(axis=1)))
            w[win] += eta * (x - w[win])
    return w


def perceptron(X, t, eta, epochs):
    """Threshold neuron y = [w0 + w1 x1 + w2 x2 >= 0]; error-correction update. Returns mistakes per epoch."""
    w = np.zeros(3)
    mistakes = []
    for _ in range(epochs):
        m = 0
        for x, target in zip(X, t):
            y = 1 if w[0] + w[1:] @ x >= 0 else 0
            err = target - y                      # 0 when correct -> no update
            m += err != 0
            w[0] += eta * err
            w[1:] += eta * err * x
        mistakes.append(int(m))
    return mistakes, w


# ================================================================ Hopfield network
class Hopfield:
    """States s_i in {-1, +1}. Storage (Hebbian): W = (1/N) sum_p xi_p xi_p^T, with W_ii = 0.
    Energy E(s) = -1/2 sum_ij W_ij s_i s_j. Asynchronous update s_i <- sign(sum_j W_ij s_j)
    can only keep E the same or lower it, so recall rolls downhill to a stored pattern."""

    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def store(self, patterns):
        for p in patterns:
            self.W += np.outer(p, p) / self.n
        np.fill_diagonal(self.W, 0.0)

    def energy(self, s):
        return float(-0.5 * s @ self.W @ s)

    def recall(self, s, sweeps, seed):
        """Update one random neuron at a time. s is changed in place. Returns the energy after every update."""
        rng = np.random.default_rng(seed)
        trace = [self.energy(s)]
        for _ in range(sweeps * self.n):
            i = rng.integers(self.n)
            s[i] = 1 if self.W[i] @ s >= 0 else -1
            trace.append(self.energy(s))
        return trace


def from_ascii(*rows):
    return np.array([1 if ch == "#" else -1 for r in rows for ch in r])


def corrupt(p, frac, seed):
    """Flip each pixel with probability frac."""
    rng = np.random.default_rng(seed)
    s = p.copy()
    flip = rng.random(len(s)) < frac
    s[flip] = -s[flip]
    return s


PATTERNS = {
    "ring nut": from_ascii("..######..", ".#......#.", "#........#", "#..####..#", "#.#....#.#",
                           "#.#....#.#", "#..####..#", "#........#", ".#......#.", "..######.."),
    "square nut": from_ascii("##########", "#........#", "#........#", "#..####..#", "#..#..#..#",
                             "#..#..#..#", "#..####..#", "#........#", "#........#", "##########"),
    "cross": from_ascii("....##....", "....##....", "....##....", "##########", "##########",
                        "....##....", "....##....", "....##....", "....##....", "....##...."),
}


def hopfield_demo(say):
    say("\n================ HOPFIELD: energy-based associative memory ================")
    names, P = list(PATTERNS), list(PATTERNS.values())
    net = Hopfield(100)
    net.store(P)
    rows, traces = [], []
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for k, (name, p) in enumerate(zip(names, P)):
        s = corrupt(p, 0.2, 10 + k)
        corrupted = s.copy()
        before = int((s != p).sum())
        tr = net.recall(s, sweeps=5, seed=20 + k)
        after = int((s != p).sum())
        never_up = bool(np.all(np.diff(tr) <= 1e-12))
        rows.append({"pattern": name, "wrong_pixels_before": before, "wrong_pixels_after": after,
                     "energy_start": round(tr[0], 3), "energy_end": round(tr[-1], 3), "energy_never_increased": never_up})
        traces.append(tr)
        say(f"{name:<11} wrong pixels {before:>2} -> {after:>2}; energy {tr[0]:.2f} -> {tr[-1]:.2f}; "
            f"energy never increased: {never_up}")
        for j, (title, img) in enumerate([("stored", p), ("corrupted 20%", corrupted), ("recalled", s)]):
            axes[k, j].imshow(img.reshape(10, 10), cmap="gray_r")
            axes[k, j].set_title(f"{name}: {title}", fontsize=9)
            axes[k, j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path("hopfield", "recall.png"), dpi=120)
    plt.close(fig)
    lines(out_path("hopfield", "energy.png"), "Energy never increases during recall", "single-neuron update",
          "energy E", names, traces)
    return rows


def run(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    summary = {}
    say("\n================ LEARNING RULES: Hebbian, Oja, competitive, perceptron ================")
    X = correlated_data(500, 30, seed=1)
    pc = pca_direction(X)
    wh, wo = np.array([0.1, 0.0]), np.array([0.1, 0.0])
    nh = hebbian(X, wh, eta=1e-4, epochs=30)
    no = oja(X, wo, eta=1e-3, epochs=30)
    summary.update({"pc1_own_svd": [round(float(v), 4) for v in pc],
                    "hebbian_final_norm": nh[-1], "hebbian_abs_cos_pc1": abs_cosine(wh, pc),
                    "oja_final_norm": no[-1], "oja_abs_cos_pc1": abs_cosine(wo, pc)})
    say(f"PC1 from our SVD: ({pc[0]:.3f}, {pc[1]:.3f})")
    say(f"Hebbian: ||w|| grew from 0.1 to {nh[-1]:.3e}; |cos(w, PC1)| = {summary['hebbian_abs_cos_pc1']:.4f}")
    say(f"Oja    : ||w|| = {no[-1]:.4f} (-> 1);      |cos(w, PC1)| = {summary['oja_abs_cos_pc1']:.4f}")
    lines(out_path("rules", "weight_norms.png"), "Weight norm: Hebbian explodes, Oja -> 1", "epoch", "||w||",
          ["Hebbian", "Oja"], [nh, no], logy=True)

    centers = np.array([[0, 0], [4, 1], [1, 4]], dtype=float)
    B, lab = blobs(100, centers, 0.6, seed=2)
    protos = competitive(B, 3, eta=0.05, epochs=20, seed=3)
    # distance from every true centre to its nearest prototype
    dist = [float(np.min(np.linalg.norm(protos - c, axis=1))) for c in centers]
    summary["competitive_prototypes"] = [[round(float(a), 2), round(float(b), 2)] for a, b in protos]
    summary["competitive_max_centre_error"] = max(dist)
    say("Competitive prototypes: " + " ".join(f"({a:.2f}, {b:.2f})" for a, b in protos)
        + f"  | true centres (0,0) (4,1) (1,4); worst distance {max(dist):.3f}")
    scatter(out_path("rules", "competitive.png"), "Winner-take-all prototypes (X)", B, lab, protos=protos)

    t = (lab != 0).astype(int)              # class 0 vs the rest: linearly separable
    mistakes, _ = perceptron(B, t, eta=0.1, epochs=15)
    summary["perceptron_mistakes_per_epoch"] = mistakes
    say(f"Perceptron (error-correction) mistakes per epoch: {mistakes}")
    lines(out_path("rules", "perceptron_mistakes.png"), "Error-correction learning", "epoch", "mistakes",
          ["mistakes"], [mistakes])

    summary["hopfield"] = hopfield_demo(say)
    return summary


if __name__ == "__main__":
    run()
