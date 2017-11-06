from collections import OrderedDict
from functools import partial
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as sp
import cPickle as pkl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from tqdm import tqdm, trange


# TODO: add support for sparse matrices
class BaseMF:
    __metaclass__ = ABCMeta

    def __init__(self, n_factors, learning_rate=0.01, reg=0.01,
                 max_iter=10, init=0.001, bias=True, verbose=False,
                 *args, **kwargs):
        """
        beta: balance of rec over side
        """
        self.lr = learning_rate
        self.reg = reg # regularization coeff

        self.k = n_factors
        self.bias = bias
        self.init = init

        self.max_iter = max_iter
        self.costs = []

        self.verbose = verbose

        # dummy variable
        self.R, self.X = None, None
        self.U, self.V, self.W = None, None, None
        self.b_u, self.b_i, self.b_w = None, None, None

    @property
    def n_users(self):
        """"""
        return self.R.shape[0]

    @property
    def n_items(self):
        """"""
        return self.R.shape[1]

    def _init_params(self, R, X=None):
        """"""
        if X is not None:
            self.X = X
            self.W = np.random.rand(X.shape[1], self.k) * self.init
            if self.bias:
                self.b_w = np.zeros((self.k,))

        # currently only supports dense mat ops.
        if sp.isspmatrix(R):
            # self.R = R.tocsr()
            self.R = R.toarray()
        else:
            if isinstance(R, np.ndarray):
                self.R = R
            else:
                raise ValueError(
                    '[ERROR] input need to be sparse or dense matrix!')

        # factors for outland
        self.U = np.zeros((self.R.shape[0], self.k))
        self.V = np.zeros((self.R.shape[1], self.k))
        if self.bias:
            self.b_u = np.zeros((self.R.shape[0],))
            self.b_i = np.zeros((self.R.shape[1],))

        # checkup zero entry row/col and make hahs for post processing
        self.row_hash = OrderedDict([(ix, j) for j, ix in
                    enumerate(np.where(self.R.sum(axis=1) > 0)[0])])
        self.col_hash = OrderedDict([(ix, j) for j, ix in
                    enumerate(np.where(self.R.sum(axis=0) > 0)[0])])

        # get squached data
        self.R_ = self.R[self.row_hash.keys()][:, self.col_hash.keys()]
        if X is not None: self.X_ = self.X[self.col_hash.keys()]

        # squashed factors for inland
        self.U_ = np.random.rand(self.R_.shape[0], self.k) * self.init
        self.V_ = np.random.rand(self.R_.shape[1], self.k) * self.init
        if self.bias:
            self.b_u_ = np.zeros((self.R_.shape[0],))
            self.b_i_ = np.zeros((self.R_.shape[1],))


    def fit(self, R, X=None):
        """"""
        self._init_params(R, X)

        if self.verbose:
            iterator = trange(self.max_iter, ncols=80)
        else:
            iterator = xrange(self.max_iter)

        for i in iterator:
            cost = self.total_cost()
            self.costs.append(cost)

            self.update_U()
            self.update_V()

            # update content-factor
            if X is not None:
                self.update_W()

            if self.verbose:
                iterator.set_description('cost {:.3f}'.format(cost))

        # post processing
        # : reassign squashed factors (trained) into outland factors
        self.U[self.row_hash.keys()] = self.U_
        self.V[self.col_hash.keys()] = self.V_
        if self.bias:
            self.b_u[self.row_hash.keys()] = self.b_u_
            self.b_i[self.col_hash.keys()] = self.b_i_

    def predict(self):
        """"""
        return self.predict_all()

    def predict_all(self):
        """"""
        if not self.bias:
            return self.U.dot(self.V.T)
        else:
            pred = self.U.dot(self.V.T)
            pred += self.b_u[:, None]
            pred += self.b_i[None, :]
            return pred

    def predict_from_item_factor(self, V_):
        """"""
        if not self.bias:
            return self.U.dot(V_.T)
        else:
            pred = self.U.dot(V_.T)
            pred += self.b_u[:, None]
            pred += self.b_i[None, :]
            return pred

    def predict_rating(self, u, i):
        """"""
        return self.U[u].dot(self.V[i]) + self.b_u[u] + self.b_i[i]

    def total_cost(self):
        """"""
        pred = self.U_.dot(self.V_.T)
        if self.bias:
            pred += self.b_u_[:, None]
            pred += self.b_i_[None, :]

        r = np.sum((self.R_ - pred)**2)
        u = self.reg * np.sum(self.U_**2)
        v = self.reg * np.sum(self.V_**2)
        cost = r + u + v

        if self.bias:
            bu = self.reg * np.sum(self.b_u_**2)
            bi = self.reg * np.sum(self.b_i_**2)
            cost += (bu + bi)
        return cost

    @abstractmethod
    def update_V(self):
        pass

    @abstractmethod
    def update_U(self):
        pass

class BaseALS(BaseMF):

    @staticmethod
    def _update_factor(ratings, other_factor, reg):
        """"""
        pass

    def update_V(self):
        """"""
        if not self.bias:
            self.V_ = self._update_factor(self.R_.T, self.U_, self.reg)
        else:
            self.V_[:], self.b_i_[:] = self._update_factor_bias(
                self.R_.T, self.U_, self.b_u_, self.reg)

    def update_U(self):
        """"""
        if not self.bias:
            self.U_ = self._update_factor(self.R_, self.V_, self.reg)
        else:
            self.U_[:], self.b_u_[:] = self._update_factor_bias(
                self.R_, self.V_, self.b_i_, self.reg)


class ExplicitALS(BaseALS):

    @staticmethod
    def _update_factor(ratings, other_factor, reg):
        """ naive implementation of ALS update
        should consume a lot of computation compared to
        other more efficient impelmentations
        """
        Y = other_factor.copy()
        YY = Y.T.dot(Y)
        k = Y.shape[1]
        X = []
        # run over items
        for r in ratings:
            A = YY + reg * np.eye(k)
            b = r.dot(Y)
            X.append(np.linalg.solve(A,b))
        return np.array(X)

    @staticmethod
    def _update_factor_bias(ratings, other_factor, other_b, reg):
        """"""
        Y_ = np.concatenate(
            [other_factor.copy(),
            np.ones((other_factor.shape[0], 1))],
            axis=1
        )
        YY = Y_.T.dot(Y_)
        ratings_ = ratings - other_b[None, :]
        k_ = Y_.shape[1]
        X_ = []
        for r in ratings_:
            A = YY + reg * np.eye(k_)
            b = r.dot(Y_)
            X_.append(np.linalg.solve(A, b))
        X_ = np.array(X_)
        return X_[:,:-1], X_[:,-1]

class ImplicitALS(BaseALS):
    """"""
    def __init__(self, n_factors, learning_rate=0.01, reg=0.1, max_iter=10,
                 alpha=1, eps=1e-5, init=0.001, bias=True, verbose=False,
                 *args, **kwargs):
        """"""
        super(ImplicitALS, self).__init__(n_factors, learning_rate,
                                          reg=reg, max_iter=max_iter,
                                          init=init, bias=bias, verbose=verbose)
        self.alpha = alpha
        self.eps = eps

    def _get_confidence_1minus(self, r):
        """"""
        return self.alpha * np.log(1. + r / self.eps)

    def _update_factor(self, ratings, other_factor, reg):
        """ naive implementation of ALS update
        should consume a lot of computation compared to
        other more efficient impelmentations
        """
        Y = sp.csr_matrix(other_factor.copy())
        confidence = self._get_confidence_1minus(ratings)
        YY = Y.T.dot(Y)
        k = Y.shape[1]
        X = []
        # run over items
        for r, c in zip(ratings, confidence):
            C = sp.diags(c, 0)

            YCm1Y = Y.T.dot(C).dot(Y)
            A = YCm1Y + YY + reg * np.eye(k)

            b = sp.csr_matrix(r).dot(C)
            b.data = b.data + 1
            b = b.dot(Y).toarray().ravel()

            X.append(np.linalg.solve(A, b))
        return np.array(X)


class ContentExplicitALS(ExplicitALS):
    """"""
    def __init__(self, n_factors, learning_rate=0.01, max_iter=10,
                 lam_u=1, lam_v=100, lam_w=1., init=0.001,
                 bias=False, verbose=False, *args, **kwargs):
        """"""
        super(ContentExplicitALS, self).__init__(n_factors, learning_rate,
                                                 reg=None, max_iter=max_iter,
                                                 init=init, bias=bias,
                                                 verbose=verbose)
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.lam_w = lam_w

    def update_U(self):
        """"""
        if not self.bias:
            self.U_ = self._update_factor(self.R_, self.V_, self.lam_u)
        else:
            self.U_[:], self.b_u_[:] = self._update_factor_bias(
                self.R_, self.V_, self.b_i_, self.lam_u)

    def update_V(self):
        """"""
        if not self.bias:
            UU = self.U_.T.dot(self.U_)
            V_ = []
            for r, x in zip(self.R_.T, self.X_):
                A = UU + self.lam_v * np.eye(self.k)
                b = r.dot(self.U_) + self.lam_v * x.dot(self.W)
                V_.append(np.linalg.solve(A, b))
            self.V_ = np.array(V_)

        else:
            U_ = np.concatenate(
                [self.U_.copy(),
                np.ones((self.U_.shape[0], 1))],
                axis=1
            )
            UU = U_.T.dot(U_)
            R_ = self.R_.T - self.b_u_[None, :]
            k_ = U_.shape[1]
            V_ = []
            for r, x, bi in zip(R_, self.X_, self.b_i_):
                A = UU + self.lam_v * np.eye(k_)
                c = x.dot(self.W) + self.b_w
                b = r.dot(U_) + self.lam_v * np.append(c, bi)
                V_.append(np.linalg.solve(A, b))
            V_ = np.array(V_)

            self.V_ = V_[:,:-1]
            self.b_i_ = V_[:,-1]

    def update_W(self):
        if not self.bias:
            d = self.X_.shape[1]
            XX = self.X_.T.dot(self.X_)
            A = self.lam_v * XX + self.lam_w * np.eye(d)
            B = self.lam_v * self.X_.T.dot(self.V_)
            self.W = np.linalg.solve(A, B)
        else:
            n, m = self.X_.shape
            X_ = np.concatenate([self.X_, np.ones((n, 1))], axis=1)

            d_ = X_.shape[1]
            XX = X_.T.dot(X_)
            A = self.lam_v * XX + self.lam_w * np.eye(d_)
            B = self.lam_v * X_.T.dot(self.V_)
            W_ = np.linalg.solve(A, B)
            self.W = W_[:-1, :]
            self.b_w = W_[-1, :]

    def total_cost(self):
        """"""
        pred = self.U_.dot(self.V_.T)
        if self.bias:
            pred += self.b_u_[:, None]
            pred += self.b_i_[None, :]

        r = np.sum((self.R_ - pred)**2)

        pred_v = self.X_.dot(self.W)
        if self.bias:
            pred_v += self.b_w

        c = self.lam_v * np.sum((self.V_ - pred_v)**2)
        u = self.lam_u * np.sum(self.U_**2)
        w = self.lam_w * np.sum(self.W**2)
        cost = r + c + u + w

        if self.bias:
            cost += self.lam_u * np.sum(self.b_u_**2)
            cost += self.lam_v * np.sum(self.b_i_**2)
            cost += self.lam_w * np.sum(self.b_w**2)

        return cost

    def predict_item_factor(self, x):
        """"""
        if not self.bias:
            return x.dot(self.W)
        else:
            return x.dot(self.W) + self.b_w


    def predict_from_side(self, x):
        """"""
        if not self.bias:
            return self.U.dot(self.predict_item_factor(x).T)
        else:
            pred = self.U.dot(self.predict_item_factor(x).T)
            pred += self.b_u[:, None]
            pred += self.b_i[None, :]
            return pred

    def predict_rating_from_side(self, u, x):
        """"""
        if not self.bias:
            return self.U[u].dot(self.predict_item_factor(x).T)
        else:
            pred = self.U[u].dot(self.predict_item_factor(x).T)
            pred += self.b_u[u]
            pred += self.b_i[None, :]
            return pred

class ContentImplicitALS(ImplicitALS):
    """"""
    def __init__(self, n_factors, learning_rate=0.01, max_iter=10,
                 lam_u=10, lam_v=10, lam_w=10, alpha=1, eps=1e-6, init=0.001,
                 bias=True, verbose=False, *args, **kwargs):
        """"""
        super(ContentImplicitALS, self).__init__(n_factors, learning_rate,
                                                 reg=None, max_iter=max_iter,
                                                 init=init, bias=bias,
                                                 alpha=alpha, eps=eps,
                                                 verbose=verbose)
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.lam_w = lam_w

    def update_U(self):
        """"""
        return self._update_factor(self.R_, self.V_, self.lam_u)

    def update_V(self):
        """"""
        C = self._get_confidence_1minus(self.R_)
        U = sp.csr_matrix(self.U_)
        UU = self.U_.T.dot(self.U_)
        V_ = []
        for r, c, x in zip(self.R_.T, C.T, self.X_):
            C = sp.diags(c)

            UCm1U = U.T.dot(C).dot(U).toarray()
            A = UCm1U + UU + self.lam_v * np.eye(self.k)

            b = sp.csr_matrix(r).dot(C)
            b.data = b.data + 1
            b = b.dot(U) + self.lam_v * x.dot(self.W)
            b = np.array(b).ravel()

            V_.append(np.linalg.solve(A, b))
        return np.array(V_)

    def update_W(self):
        d = self.X_.shape[1]
        XX = self.X_.T.dot(self.X_)
        A = self.lam_v * XX + self.lam_w * np.eye(d)
        B = self.lam_v * self.X.T.dot(self.V_)
        return np.linalg.solve(A, B)

    def total_cost(self):
        """"""
        r = np.sum((self.R_ - self.U_.dot(self.V_.T))**2)
        C = self.V_ - self.X_.dot(self.W)
        c = self.lam_v * np.sum(C**2)
        u = self.lam_u * np.sum(U_**2)
        w = self.lam_w * np.sum(W**2)
        return r + c + u + w

    def predict_item_factor(self, x):
        """"""
        return x.dot(self.W)

    def predict_from_side(self, x):
        """"""
        return self.U.dot(self.predict_item_factor(x).T)

    def predict_rating_from_side(self, u, x):
        """"""
        return self.U[u].dot(self.predict_item_factor(x).T)



if __name__ == "__main__":
    import h5py
    hf = h5py.File('/Users/jaykim/Downloads/data/feature50/individual/conv_2d_artist_50_0.0001_ThisIsMyJam_feature.h5')
    R = hf['y'][:].T
    X = hf['X'][:]
    rnd_u = np.random.choice(R.shape[0], R.shape[0], replace=False)
    rnd_t = np.random.choice(R.shape[1], R.shape[1], replace=False)
    R = R[rnd_u][:, rnd_t]
    X = X[rnd_t]

    # model = ExplicitALS(n_factors=10, bias=True, max_iter=15, verbose=True)
    # model.fit(R)
    
    model = ContentExplicitALS(n_factors=50, bias=True, max_iter=15, verbose=True)
    model.fit(R,X)
