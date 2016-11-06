# coding:utf-8
import numpy as np
from sklearn.decomposition import SparseCoder, DictionaryLearning
from transform_learning import MiniBatchTransformLearning


def test_mini_batch():
    np.random.seed(0)

    T = 100
    # init
    N = 16
    M = 20
    Y = np.random.randn(M * T, N)
    K = 4

    tl = MiniBatchTransformLearning(transform_n_nonzero_coefs=K)
    for i in range(T):
        ys = Y[i * M:(i + 1) * M]
        tl.partial_fit(ys)

    sc = SparseCoder(tl.components_, transform_n_nonzero_coefs=K)
    code = sc.fit_transform(Y)
    tl_error = np.linalg.norm(Y - code.dot(tl.components_))
    print('Mini Batch Transform Learning:', tl_error)

    W = np.random.randn(N, N)
    sc = SparseCoder(W, transform_n_nonzero_coefs=K)
    code = sc.fit_transform(Y)
    random_error = np.linalg.norm(Y - code.dot(W))
    print('Random Dictionary:', random_error)
    assert tl_error < random_error
