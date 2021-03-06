import os
import sys
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# TODO: currently installation of numba-plsa must be done manually
#       can automate?
sys.path.append(os.path.join(DIR_PATH,'../../../numba-plsa/'))

# TODO: currently installation of wrmf must be done manually
#       can automate?
sys.path.append(os.path.join(DIR_PATH,'../../../wmf/'))

from sklearn.decomposition import LatentDirichletAllocation as LDA
from numba_plsa.plsa import plsa
from wmf import log_surplus_confidence_matrix, factorize, recompute_factors

# =====================================================================
# classes for wraping sub level processes (MF, GMM, etc..)
# =====================================================================
class MatrixFactorization:
    """"""
    def __init__(self, k, n_iter=100, alg='plsa'):
        """"""
        self.k = k

        # assign main function
        if alg == 'plsa':
            self.alg = PLSA(self.k, n_iter=n_iter)
        elif alg == 'lda':
            self.alg = LDA(self.k, max_iter=n_iter, n_jobs=-1)
        elif alg == 'wrmf':
            self.alg = WRMF(self.k, n_iter=n_iter)
        else:
            raise ValueError(
                '[ERROR] {} is not supported algorithm!'.format(alg)
            )

    @property
    def components_(self):
        """"""
        return self.alg.components_

    def fit_transform(self, X):
        """"""
        return self.alg.fit_transform(X)


class PLSA:
    """ Simple wrapper object for sklearn-style interface for numba-plsa """
    def __init__(self, n_components, min_count=1, n_iter=100, method='numba'):
        """"""
        self.k = n_components
        self.min_count = min_count
        self.n_iter = n_iter
        self.method = method

    def fit_transform(self, X):
        """"""
        transformed, self.components_ = plsa(
            X, self.k, self.n_iter, self.min_count, self.method)

        return transformed


class WRMF:
    """ Simple wrapper object for sklearn-style interface for wmf (benanne) """
    def __init__(
        self, n_components, alpha=40, epsilon=1e-1, lambda_reg=1e-5,
        n_iter=20, init_std=0.01, verbose=False, *args, **kwargs):
        """"""
        self.k = n_components
        self.lambda_reg = lambda_reg
        self.n_iter = n_iter
        self.init_std = init_std
        self.verbose = verbose

        self.alpha = alpha
        self.epsilon = epsilon

    def fit_transform(self, X):
        """"""
        S = log_surplus_confidence_matrix(X, self.alpha, self.epsilon)
        transformed, self.components_ = factorize(
            S, self.k, self.lambda_reg, self.n_ter,
            self.init_std, self.verbose
        )
        return transformed
