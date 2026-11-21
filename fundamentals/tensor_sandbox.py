"""Requirement 3: maths sandbox.

1. A small N-dimensional Tensor class written by hand: shape, strides, reshape (with -1),
   broadcasting (NumPy's right-alignment rule), dot, matmul, transpose.
2. SVD by one-sided Jacobi rotations, checked against np.linalg.svd, and used for a
   low-rank approximation of a 64x64 image.
3. Numerical gradient check (central difference) and how its error depends on the step h.

NumPy is used only as a flat number container and for simple vector arithmetic;
the algorithms (index arithmetic, broadcasting rule, Jacobi rotations) are written here.
"""
import math

import matplotlib.pyplot as plt
import numpy as np

from fundamentals.common import lines, out_path


# ================================================================ Tensor
def numel(shape):
    """Number of elements = product of the dimensions."""
    n = 1
    for s in shape:
        n *= s
    return n


def compute_strides(shape):
    """Row-major strides: how many flat positions to jump when index d increases by 1.
    shape (2, 3) -> strides (3, 1): element (i, j) lives at i*3 + j*1."""
    strides, acc = [0] * len(shape), 1
    for d in range(len(shape) - 1, -1, -1):
        strides[d] = acc
        acc *= shape[d]
    return tuple(strides)


def broadcast_shape(a, b):
    """Broadcasting rule (same as NumPy): align the shapes from the RIGHT; two dimensions are
    compatible when they are equal or one of them is 1. Missing leading dimensions count as 1.
    (3,1) & (1,4) -> (3,4);  (2,3) & (3,) -> (2,3);  (2,3) & (2,) -> error."""
    n = max(len(a), len(b))
    pa = (1,) * (n - len(a)) + tuple(a)           # pad the shorter shape with 1s on the LEFT
    pb = (1,) * (n - len(b)) + tuple(b)
    out = []
    for da, db in zip(pa, pb):
        if da != db and da != 1 and db != 1:
            raise ValueError(f"shapes {tuple(a)} and {tuple(b)} are not broadcastable")
        out.append(max(da, db))
    return tuple(out)


class Tensor:
    """Minimal N-dimensional tensor: a flat array of numbers + shape + strides."""

    def __init__(self, data, shape):
        data = np.asarray(data, dtype=float).ravel()
        shape = tuple(int(s) for s in shape)
        if data.size != numel(shape):
            raise ValueError(f"data has {data.size} values but shape {shape} needs {numel(shape)}")
        self.data = data                      # flat storage
        self.shape = shape
        self.strides = compute_strides(shape)

    # ---------- factories
    @staticmethod
    def arange(n):
        return Tensor(range(n), (n,))

    @staticmethod
    def from_rows(rows):
        return Tensor([v for r in rows for v in r], (len(rows), len(rows[0])))

    # ---------- element access
    def offset(self, idx):
        """Flat position of a multi-index: sum_d idx[d] * strides[d]."""
        if len(idx) != len(self.shape):
            raise IndexError("wrong number of indices")
        o = 0
        for i, s, st in zip(idx, self.shape, self.strides):
            if not 0 <= i < s:
                raise IndexError(f"index {idx} out of range for shape {self.shape}")
            o += i * st
        return o

    def get(self, *idx):
        return float(self.data[self.offset(idx)])

    # ---------- reshape / transpose
    def reshape(self, *new_shape):
        """Same data, new shape (a view: no copy). One dimension may be -1 (inferred)."""
        s = list(new_shape)
        if s.count(-1) > 1:
            raise ValueError("only one dimension can be -1")
        if -1 in s:
            known = numel([d for d in s if d != -1])
            if known == 0 or self.data.size % known:
                raise ValueError(f"cannot reshape {self.shape} to {tuple(new_shape)}")
            s[s.index(-1)] = self.data.size // known
        if numel(s) != self.data.size:
            raise ValueError(f"cannot reshape {self.shape} to {tuple(new_shape)}")
        t = Tensor.__new__(Tensor)            # share the same flat data
        t.data, t.shape, t.strides = self.data, tuple(s), compute_strides(s)
        return t

    def transpose(self):
        """2-D transpose: element (i, j) moves to (j, i). Copies the data."""
        self._check_2d()
        r, c = self.shape
        out = np.zeros(r * c)
        for i in range(r):
            for j in range(c):
                out[j * r + i] = self.data[i * c + j]
        return Tensor(out, (c, r))

    # ---------- broadcasting element-wise operations
    def _broadcast(self, other, op):
        if not isinstance(other, Tensor):
            other = Tensor([other], (1,))
        out_shape = broadcast_shape(self.shape, other.shape)
        out_strides = compute_strides(out_shape)
        flat = np.arange(numel(out_shape))

        def source_offsets(t):
            """For every output position, where to read in t. A size-1 dimension is 'stretched':
            it always uses index 0, so its contribution to the offset is 0."""
            shift = len(out_shape) - len(t.shape)
            off = np.zeros(flat.size, dtype=int)
            for d, size in enumerate(t.shape):
                if size == 1:
                    continue
                od = d + shift
                index_d = (flat // out_strides[od]) % out_shape[od]
                off += index_d * t.strides[d]
            return off

        values = op(self.data[source_offsets(self)], other.data[source_offsets(other)])
        return Tensor(values, out_shape)

    def __add__(self, o): return self._broadcast(o, lambda x, y: x + y)
    def __sub__(self, o): return self._broadcast(o, lambda x, y: x - y)
    def __mul__(self, o): return self._broadcast(o, lambda x, y: x * y)

    # ---------- inner products
    @staticmethod
    def dot(a, b):
        """Inner product: sum_i a_i * b_i (same number of elements)."""
        if a.data.size != b.data.size:
            raise ValueError("dot: size mismatch")
        s = 0.0
        for x, y in zip(a.data, b.data):
            s += x * y
        return s

    def matmul(self, o):
        """(m,k) @ (k,n) -> (m,n). Each output = inner product of a row of self and a column of o."""
        self._check_2d()
        o._check_2d()
        m, k = self.shape
        k2, n = o.shape
        if k != k2:
            raise ValueError(f"matmul: {self.shape} @ {o.shape}")
        out = np.zeros(m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += self.data[i * k + p] * o.data[p * n + j]
                out[i * n + j] = s
        return Tensor(out, (m, n))

    def norm(self):
        return math.sqrt(Tensor.dot(self, self))

    def numpy(self):
        return self.data.reshape(self.shape).copy()

    def _check_2d(self):
        if len(self.shape) != 2:
            raise ValueError(f"expected a 2-D tensor, got shape {self.shape}")

    def __repr__(self):
        if len(self.shape) == 2:
            rows = ["  [" + " ".join(f"{self.get(i, j):7.2f}" for j in range(self.shape[1])) + " ]"
                    for i in range(self.shape[0])]
            return f"Tensor{self.shape}\n" + "\n".join(rows)
        return f"Tensor{self.shape} {self.data.tolist()}"


# ================================================================ SVD (one-sided Jacobi)
def svd_jacobi(A, tol=1e-12, max_sweeps=100):
    """A (m x n, m >= n) = U diag(S) V^T.

    Keep rotating PAIRS of columns of A until every pair is orthogonal; apply the same
    rotations to V (which starts as the identity). Then:
      column norms of the rotated A = singular values S
      rotated A / S                 = U
      accumulated rotations         = V
    Returns U (m x n), S (n,), V (n x n), sorted by decreasing singular value."""
    a = np.array(A, dtype=float)
    m, n = a.shape
    if m < n:
        raise ValueError("this simple SVD expects rows >= cols; transpose first")
    v = np.eye(n)
    for _ in range(max_sweeps):
        off = 0.0                                   # largest |cos(angle)| between two columns
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = a[:, p] @ a[:, p]           # ||a_p||^2
                beta = a[:, q] @ a[:, q]            # ||a_q||^2
                gamma = a[:, p] @ a[:, q]           # a_p . a_q
                if abs(gamma) < 1e-15:
                    continue
                off = max(off, abs(gamma) / math.sqrt(alpha * beta))
                # rotation angle that makes columns p and q orthogonal
                zeta = (beta - alpha) / (2 * gamma)
                t = 1.0 if zeta == 0 else math.copysign(1, zeta) / (abs(zeta) + math.sqrt(1 + zeta * zeta))
                c = 1 / math.sqrt(1 + t * t)
                s = c * t
                ap, aq = a[:, p].copy(), a[:, q].copy()
                a[:, p], a[:, q] = c * ap - s * aq, s * ap + c * aq
                vp, vq = v[:, p].copy(), v[:, q].copy()
                v[:, p], v[:, q] = c * vp - s * vq, s * vp + c * vq
        if off < tol:
            break
    sv = np.sqrt((a * a).sum(axis=0))
    order = np.argsort(-sv)
    S = sv[order]
    U = np.where(S > 1e-15, a[:, order] / np.where(S > 1e-15, S, 1), 0.0)
    V = v[:, order]
    return U, S, V


def reconstruct(U, S, V, k):
    """Low-rank approximation: keep only the top-k singular values."""
    return (U[:, :k] * S[:k]) @ V[:, :k].T


def frob(A):
    """Frobenius norm: square root of the sum of all squared entries."""
    return math.sqrt(float((np.asarray(A) ** 2).sum()))


# ================================================================ gradient check
def numerical_grad(param, loss_fn, h=1e-5):
    """Central difference for every entry of param (changed in place, then restored):
    dL/dθ_i ≈ [L(θ_i + h) - L(θ_i - h)] / (2h). The error is O(h^2)."""
    g = np.zeros_like(param)
    flat, gflat = param.reshape(-1), g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        lp = loss_fn()
        flat[i] = old - h
        lm = loss_fn()
        flat[i] = old
        gflat[i] = (lp - lm) / (2 * h)
    return g


def max_relative_error(numeric, analytic):
    """max |a - b| / (|a| + |b|). Rule of thumb: < 1e-7 excellent, < 1e-5 fine, > 1e-3 a bug."""
    a, b = np.ravel(numeric), np.ravel(analytic)
    denom = np.abs(a) + np.abs(b)
    rel = np.where(denom < 1e-12, 0.0, np.abs(a - b) / np.maximum(denom, 1e-12))
    return float(rel.max())


# ================================================================ experiment
def surface_image(n=64, seed=1):
    """Synthetic 64x64 'surface': a horizontal sine texture + a nut-like ring + noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ring = 1.0 if abs(math.hypot(i - 32, j - 32) - 20) < 3 else 0.0
            img[i, j] = 0.5 * math.sin(j / 6.0) + ring + 0.1 * rng.standard_normal()
    return img


def run(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    summary = {}
    say("\n================ SANDBOX: tensors, broadcasting, reshape, SVD, gradient check ================")

    a = Tensor.arange(6).reshape(2, 3)
    say("a = arange(6).reshape(2,3):\n", a)
    say("a.reshape(3,-1).shape =", a.reshape(3, -1).shape)
    col, row = Tensor([10, 20], (2, 1)), Tensor([1, 2, 3], (3,))
    say("broadcast (2,3)+(2,1):\n", a + col)
    say("broadcast (2,3)*(3,):\n", a * row)
    say("broadcast (2,1)+(3,) -> outer sum (2,3):\n", col + row)
    try:
        a + Tensor([1, 2], (2,))
    except ValueError as e:
        say("expected error:", e)
        summary["broadcast_error_message"] = str(e)
    u, v = Tensor([1, 2, 3], (3,)), Tensor([4, 5, 6], (3,))
    summary["dot_u_v"] = Tensor.dot(u, v)
    say(f"dot(u,v) = {summary['dot_u_v']}   (1*4 + 2*5 + 3*6 = 32)")
    say("a @ a^T =\n", a.matmul(a.transpose()))

    # SVD of a small matrix, compared with NumPy
    M = np.array([[3, 2], [2, 3], [2, -2]], dtype=float)
    U, S, V = svd_jacobi(M)
    S_np = np.linalg.svd(M, compute_uv=False)
    summary["svd_small_singular_values"] = [round(float(s), 6) for s in S]
    summary["svd_small_reconstruction_error"] = frob(M - reconstruct(U, S, V, 2))
    summary["svd_small_max_diff_vs_numpy"] = float(np.abs(S - S_np).max())
    say("SVD of 3x2 M: singular values =", summary["svd_small_singular_values"],
        "| numpy:", [round(float(s), 6) for s in S_np])
    say(f"||M - U S V^T|| = {summary['svd_small_reconstruction_error']:.2e}")

    # low-rank approximation of a 64x64 image
    img = surface_image()
    U, S, V = svd_jacobi(img)
    S_np = np.linalg.svd(img, compute_uv=False)
    summary["svd_64_max_diff_vs_numpy"] = float(np.abs(S - S_np).max())
    errs = np.array([frob(img - reconstruct(U, S, V, k)) / frob(img) for k in range(1, 65)])
    summary["svd_64_rel_error_rank1"] = float(errs[0])
    summary["svd_64_rel_error_rank5"] = float(errs[4])
    summary["svd_64_rel_error_rank20"] = float(errs[19])
    say(f"64x64 image: max |S_ours - S_numpy| = {summary['svd_64_max_diff_vs_numpy']:.2e}")
    say(f"64x64 image: relative error rank-1 = {errs[0]:.3f}, rank-5 = {errs[4]:.3f}, rank-20 = {errs[19]:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (title, M_) in zip(axes, [("original", img)] +
                               [(f"rank {k} (error {errs[k - 1]:.2f})", reconstruct(U, S, V, k)) for k in (1, 5, 20)]):
        ax.imshow(M_, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path("sandbox", "svd_reconstructions.png"), dpi=120)
    plt.close(fig)
    lines(out_path("sandbox", "singular_values.png"), "Singular values of the 64x64 image", "index", "sigma",
          ["ours (Jacobi)", "np.linalg.svd"], [S, S_np], logy=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.arange(1, 65), errs, "o-", markersize=3)
    ax.set(title="Relative reconstruction error vs rank", xlabel="rank k", ylabel="||A - A_k|| / ||A||")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path("sandbox", "svd_error.png"), dpi=120)
    plt.close(fig)

    # gradient check on f(x) = sum(sin(x) * x^2),  f'(x) = cos(x) x^2 + 2 x sin(x)
    x = np.random.default_rng(0).standard_normal(5)
    f = lambda: float(np.sum(np.sin(x) * x * x))
    analytic = np.cos(x) * x * x + 2 * x * np.sin(x)
    summary["gradcheck_rel_error_h1e-5"] = max_relative_error(numerical_grad(x, f, 1e-5), analytic)
    say(f"Gradient check f = sum(sin(x) x^2): max relative error = {summary['gradcheck_rel_error_h1e-5']:.2e}")
    hs = [10.0 ** -(k + 1) for k in range(12)]
    errs_h = [max_relative_error(numerical_grad(x, f, h), analytic) for h in hs]
    best = int(np.argmin(errs_h))
    summary["gradcheck_best_h"] = hs[best]
    summary["gradcheck_best_error"] = errs_h[best]
    say(f"Best step h = {hs[best]:.0e} (error {errs_h[best]:.2e}); too large h -> truncation error, "
        "too small -> round-off")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.loglog(hs, errs_h, "o-")
    ax.set(title="Gradient-check error vs step h", xlabel="h", ylabel="max relative error")
    ax.invert_xaxis()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path("sandbox", "gradcheck_vs_h.png"), dpi=120)
    plt.close(fig)
    say("Figures written to experiments/fundamentals/sandbox/")
    return summary


if __name__ == "__main__":
    run()
