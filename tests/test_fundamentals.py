"""Tests for the NumPy fundamentals package.   Run:  python -m pytest -q"""
import numpy as np
import pytest

from fundamentals.common import xor_data
from fundamentals.learning_rules import PATTERNS, Hopfield, abs_cosine, correlated_data, corrupt, oja, pca_direction
from fundamentals.mlp import MLP, Optimizer, gradient_check, train
from fundamentals.tensor_sandbox import Tensor, broadcast_shape, reconstruct, svd_jacobi


# ---------------------------------------------------------------- Tensor
@pytest.mark.parametrize("sa, sb", [((2, 3), (2, 1)), ((2, 3), (3,)), ((2, 1), (3,)), ((3, 1), (1, 4)),
                                    ((2, 1, 4), (3, 1)), ((5,), (1,))])
def test_broadcasting_matches_numpy(sa, sb):
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(sa), rng.standard_normal(sb)
    out = Tensor(a, sa) + Tensor(b, sb)
    assert out.shape == np.broadcast_shapes(sa, sb)
    np.testing.assert_allclose(out.numpy(), a + b)
    np.testing.assert_allclose((Tensor(a, sa) * Tensor(b, sb)).numpy(), a * b)


@pytest.mark.parametrize("sa, sb", [((2, 3), (2,)), ((4, 3), (2, 3)), ((2, 3, 4), (3, 3))])
def test_broadcasting_incompatible_raises(sa, sb):
    with pytest.raises(ValueError):
        broadcast_shape(sa, sb)


def test_reshape():
    t = Tensor.arange(12)
    assert t.reshape(3, -1).shape == (3, 4)
    assert t.reshape(-1, 2, 3).shape == (2, 2, 3)
    assert t.reshape(3, 4).get(2, 1) == 9                 # row-major: 2*4 + 1
    assert t.reshape(3, 4).strides == (4, 1)
    with pytest.raises(ValueError):
        t.reshape(5, -1)                                   # 12 is not divisible by 5
    with pytest.raises(ValueError):
        t.reshape(-1, -1)


def test_dot_matmul_transpose():
    assert Tensor.dot(Tensor([1, 2, 3], (3,)), Tensor([4, 5, 6], (3,))) == 32
    a = np.arange(6.0).reshape(2, 3)
    A = Tensor(a, (2, 3))
    np.testing.assert_allclose(A.transpose().numpy(), a.T)
    np.testing.assert_allclose(A.matmul(A.transpose()).numpy(), a @ a.T)


# ---------------------------------------------------------------- SVD
def test_svd_reconstruction_and_singular_values():
    A = np.random.default_rng(1).standard_normal((8, 5))
    U, S, V = svd_jacobi(A)
    np.testing.assert_allclose(reconstruct(U, S, V, 5), A, atol=1e-10)
    np.testing.assert_allclose(S, np.linalg.svd(A, compute_uv=False), atol=1e-10)
    np.testing.assert_allclose(U.T @ U, np.eye(5), atol=1e-10)     # orthonormal columns
    np.testing.assert_allclose(V.T @ V, np.eye(5), atol=1e-10)


# ---------------------------------------------------------------- MLP
@pytest.mark.parametrize("act, l2", [("tanh", 0.0), ("sigmoid", 0.01), ("tanh", 0.1)])
def test_backprop_gradient_check(act, l2):
    X = np.random.default_rng(2).standard_normal((6, 3))
    Y = np.array([[0], [1], [1], [0], [1], [0]], dtype=float)
    net = MLP([3, 5, 4, 1], act=act, init="xavier", seed=4, l2=l2)
    assert gradient_check(net, X, Y) < 1e-6


def test_xor_is_learned():
    X, Y = xor_data()
    net = MLP([2, 4, 1], act="tanh", init="xavier", seed=7)
    train(net, Optimizer("gd", lr=0.5), (X, Y), epochs=2000, batch_size=4, seed=1)
    assert net.accuracy(X, Y) == 1.0


# ---------------------------------------------------------------- learning rules
def test_oja_finds_first_principal_component():
    X = correlated_data(500, 30, seed=1)
    w = np.array([0.1, 0.0])
    norms = oja(X, w, eta=1e-3, epochs=30)
    assert abs_cosine(w, pca_direction(X)) > 0.99
    assert abs(norms[-1] - 1) < 0.05                       # Oja keeps ||w|| near 1


def test_hopfield_energy_never_increases():
    net = Hopfield(100)
    net.store(list(PATTERNS.values()))
    for k, p in enumerate(PATTERNS.values()):
        s = corrupt(p, 0.2, seed=10 + k)
        trace = net.recall(s, sweeps=5, seed=20 + k)
        assert np.all(np.diff(trace) <= 1e-12)
