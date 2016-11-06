# coding:utf-8
import numpy as np


class MiniBatchTransformLearning(object):

    def __init__(self, transform_n_nonzero_coefs):
        self.J = 0
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.initialized = False

    def initialize(self, N):
        W = np.random.randn(N, N)
        self.components_ = W / np.linalg.norm(W, axis=1)[:, np.newaxis]
        self.beta = 0
        self.theta = self.gamma = np.zeros((N, N))
        self.initialized = True

    def partial_fit(self, ys):
        M, N = ys.shape
        if not self.initialized:
            self.initialize(N)

        self.J += 1
        # sparse coding
        K = self.transform_n_nonzero_coefs
        X = self.components_.dot(ys.T).T
        thres = -np.partition(-np.abs(X), K, axis=1)[:, K:K + 1]
        X[X < thres] = 0

        # prepare
        lmd = 1.0
        theta, gamma, beta = self.theta, self.gamma, self.beta
        a = 1. / self.J
        theta = (1. - a) * theta + a * 1. / M * ys.T.dot(X)
        gamma = (1. - a) * gamma + a * 1. / M * ys.T.dot(ys)
        beta = (1. - a) * beta + a * 1. / M * lmd * np.linalg.norm(ys) ** 2
        self.theta, self.gamma, self.beta = theta, gamma, beta

        # update
        U, delta, U_t = np.linalg.svd(gamma + beta * np.eye(N))
        L = U.dot(np.diag(np.power(delta, 0.5))).dot(U_t)
        L_inv = np.linalg.inv(L)
        Q, sigma, R = np.linalg.svd(L_inv.dot(theta))
        sigma = np.diag(sigma)
        tmp = np.power(np.power(sigma, 2) + 2 * beta * np.eye(N), 0.5)
        self.components_ = 0.5 * R.dot(sigma + tmp).dot(Q.T).dot(L_inv)

        return self
