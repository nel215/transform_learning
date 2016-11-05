# coding:utf-8
import numpy as np


def u(U, S, V, A):
    M = U.dot(S).dot(V.T) + A
    U_, S_, V_ = np.linalg.svd(M)
    return U_, np.diag(S_), V_.T


def Hs(x, s):
    a = np.abs(x)
    n = -np.partition(-a, s)
    x[a <= n[s]] = 0
    return x


class OnlineTransformLearning(object):

    def __init__(self, transform_n_nonzero_coefs):
        self.t = 0
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def fit(self, X):
        lmd = 1.0
        N = X.shape[1]
        W = np.random.randn(N, N)
        W /= np.linalg.norm(W, axis=1)[:, np.newaxis]
        beta = delta = 0
        delta, sigma = np.zeros((N, N)), np.zeros((N, N))
        U = Q = R = np.eye(N)
        for y in X:
            self.t += 1
            alpha = 1. / self.t

            # sparse coding
            x = W.dot(y)
            x = Hs(x, self.transform_n_nonzero_coefs)

            # update beta
            beta = (1. - alpha) * beta + alpha * lmd * np.linalg.norm(y) ** 2

            # transform
            # a
            A = alpha * y[:, np.newaxis].dot(y[:, np.newaxis].T)
            U, delta, _ = u(U, (1. - alpha) * delta, U, A)

            # b
            tmp = np.power(delta + beta * np.eye(N), 0.5)
            L = U.dot(tmp).dot(U.T)
            L_inv = np.linalg.inv(L)

            # c
            A_hat = alpha * L_inv.dot(y[:, np.newaxis]).dot(x[:, np.newaxis].T)
            Q, sigma, R = u(Q, (1. - alpha) * sigma, R, A_hat)

            # d
            tmp = np.sqrt(np.power(sigma, 2) + 2. * beta * np.eye(N))
            W = 0.5 * R.dot(sigma + tmp).dot(Q.T).dot(L_inv)

        self.components_ = W

        return self
