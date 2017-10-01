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
        self.b = bias
        self.init = init

        self.max_iter = max_iter
        self.costs = []

        self.verbose = verbose

        # dummy variable
        self.R, self.X = None, None
        self.U, self.V, self.W = None, None, None

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

            self.U_ = self.update_U()
            self.V_ = self.update_V()

            # update content-factor
            if X is not None:
                self.W = self.update_W()

            if self.verbose:
                iterator.set_description('cost {:.3f}'.format(cost))

        # post processing
        # : reassign squashed factors (trained) into outland factors
        self.U[self.row_hash.keys()] = self.U_
        self.V[self.col_hash.keys()] = self.V_

    def predict(self):
        """"""
        return self.predict_all()

    def predict_all(self):
        """"""
        return self.U.dot(self.V.T)

    def predict_rating(self, u, i):
        """"""
        return self.U[u].dot(self.V[i])

    def total_cost(self):
        """"""
        r = self.R_ - self.U_.dot(self.V_.T)
        u = self.reg * np.sum(self.U_**2)
        v = self.reg * np.sum(self.V_**2)
        return r + u + v

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
        return self._update_factor(self.R_.T, self.U_, self.reg)

    def update_U(self):
        """"""
        return self._update_factor(self.R_, self.V_, self.reg)

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
                 lam_u=0.1, lam_v=10, lam_w=1., init=0.001,
                 bias=True, verbose=False, *args, **kwargs):
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
        return self._update_factor(self.R_, self.V_, self.lam_u)

    def update_V(self):
        """"""
        UU = self.U_.T.dot(self.U_)
        V_ = []
        for r, x in zip(self.R_.T, self.X_):
            A = UU + self.lam_v * np.eye(self.k)
            b = r.dot(self.U_) + self.lam_v * x.dot(self.W)
            V_.append(np.linalg.solve(A, b))
        return np.array(V_)

    def update_W(self):
        d = self.X_.shape[1]
        XX = self.X_.T.dot(self.X_)
        A = self.lam_v * XX + self.lam_w * np.eye(d)
        B = self.lam_v * self.X_.T.dot(self.V_)
        return np.linalg.solve(A, B)

    def total_cost(self):
        """"""
        r = np.sum((self.R_ - self.U_.dot(self.V_.T))**2)
        C = self.V_ - self.X_.dot(self.W)
        c = self.lam_v * np.sum(C**2)
        u = self.lam_u * np.sum(self.U_**2)
        w = self.lam_w * np.sum(self.W**2)
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


class ContentImplicitALS(ImplicitALS):
    """"""
    def __init__(self, n_factors, learning_rate=0.01, max_iter=10,
                 lam_u=0.1, lam_v=10, lam_w=0.1, alpha=1, eps=1e-6, init=0.001,
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

