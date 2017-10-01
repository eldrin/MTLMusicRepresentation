import numpy as np
from scipy import sparse as sp
import cPickle as pkl

from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from tqdm import tqdm, trange


# TODO: add support for sparse matrices
class BaseMF:
    __metaclass__ = ABCMeta

    def __init__(self, n_factors, learning_rate=0.01, alpha=0.01, beta=0.5,
                 max_iter=10, init=0.001, bias=True, verbose=False,
                 *args, **kwargs):
        """
        beta: balance of rec over side
        """
        self.lr = learning_rate
        self.alpha = alpha # regularization coeff
        self.beta = beta # "rec to side" balance [0, 1]
        if self.beta < 0 or self.beta > 1:
            raise ValueError(
                '[ERROR] lambda should be in [0, 1)')

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
        self.U = np.random.rand(self.R.shape[0], self.k) * self.init
        self.V = np.random.rand(self.R.shape[1], self.k) * self.init

    def fit(self, R, X=None):
        """"""
        self._init_params(R, X)

        if self.verbose:
            iterator = trange(self.max_iter, ncols=80)
        else:
            iterator = xrange(self.max_iter)

        for i in iterator:
            c_rec, c_con = self.validate()
            self.costs.append((c_rec, c_con))

            self.U = self.update_U()
            self.V = self.update_V()

            # update content-factor
            if X is not None:
                self.W = self.update_W()

            if self.verbose:
                if X is not None:
                    iterator.set_description(
                        'cost {:.3f} | {:.3f}'.format(c_rec, c_con))
                else:
                    iterator.set_description(
                        'cost {:.3f}'.format(c_rec))

    def predict(self):
        """"""
        return self.predict_all()

    def predict_all(self):
        """"""
        return self.U.dot(self.V.T)

    def predict_rating(self, u, i):
        """"""
        return self.U[u].dot(self.V[i])

    def error(self, typ='rec'):
        """"""
        if self.R is None:
            raise ValueError('[ERROR] No interaction data!')

        if typ == 'rec':
            return self.R - self.U.dot(self.V.T)
        elif typ == 'con':
            if self.X is None:
                # raise ValueError('[ERROR] No feature data!')
                return None
            else:
                return self.V - self.X.dot(self.W)

    def cost(self, typ='rec'):
        """"""
        e = self.error(typ)
        if e is not None:
            return np.sqrt(np.mean(np.square(e)))
        else:
            return e

    def validate(self):
        """"""
        c_rec = self.cost('rec')
        c_con = self.cost('con')
        return c_rec, c_con

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
        return self._update_factor(self.R.T, self.U, self.alpha)

    def update_U(self):
        """"""
        return self._update_factor(self.R, self.V, self.alpha)

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

class ContentExplicitALS(BaseALS):

    @property
    def lam_r(self):
        return self.beta

    @property
    def lam_c(self):
        return 1. - self.beta

    def update_U(self):
        """"""
        VV = self.V.T.dot(self.V)
        U_ = []
        for r in self.R:
            A = self.lam_r * VV + self.alpha * np.eye(self.k)
            b = self.lam_r * r.dot(self.V)
            U_.append(np.linalg.solve(A, b))
        return np.array(U_)

    def update_V(self):
        """"""
        UU = self.U.T.dot(self.U)
        V_ = []
        for r, x in zip(self.R.T, self.X):
            A = self.lam_r * UU + self.lam_c * np.eye(self.k)
            b = self.lam_r * r.dot(self.U) + self.lam_c * x.dot(self.W)
            V_.append(np.linalg.solve(A, b))
        return np.array(V_)

    def update_W(self):
        d = self.X.shape[1]
        XX = self.X.T.dot(self.X)
        A = self.lam_c * XX + self.alpha * np.eye(d)
        b = self.lam_c * self.X.T.dot(self.V)
        return np.linalg.solve(A, b)

    def predict_item_factor(self, x):
        """"""
        return x.dot(self.W)

    def predict_from_side(self, x):
        """"""
        return self.U.dot(self.predict_item_factor(x).T)

    def predict_rating_from_side(self, u, x):
        """"""
        return self.U[u].dot(self.predict_item_factor(x).T)
