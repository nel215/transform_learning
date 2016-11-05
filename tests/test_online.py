# coding:utf-8
import numpy as np
from sklearn.decomposition import SparseCoder, DictionaryLearning
from transform_learning import OnlineTransformLearning


def test_fit():
    np.random.seed(0)
    N = 32
    K = 4

    X = np.random.randn(100, N)
    X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

    tl = OnlineTransformLearning(transform_n_nonzero_coefs=K).fit(X)
    sc = SparseCoder(tl.components_, transform_n_nonzero_coefs=K)
    code = sc.fit_transform(X)
    tl_error = np.linalg.norm(X - code.dot(tl.components_))
    print('Online Transform Learning:', tl_error)

    W = np.random.randn(N, N)
    sc = SparseCoder(W, transform_n_nonzero_coefs=K)
    code = sc.fit_transform(X)
    random_error = np.linalg.norm(X - code.dot(tl.components_))
    print('Random Dictionary:', random_error)
    assert tl_error < random_error
