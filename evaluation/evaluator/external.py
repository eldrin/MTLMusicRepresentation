import time
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as sp
# from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (cross_val_predict,
                                     StratifiedKFold,
                                     KFold,
                                     GridSearchCV)
from sklearn.metrics import (classification_report,
                             accuracy_score,
                             r2_score,
                             confusion_matrix)
from ml_metrics.average_precision import apk

from model.recsys.model import ContentExplicitALS
# from model.recsys.model import ExplicitALS
import h5py
from tqdm import tqdm


class BaseExternalTaskEvaluator(object):
    __metaclass__ = ABCMeta

    """"""
    def __init__(self, fns, preproc=None, n_jobs=-1):
        """"""
        self.data = []
        for fn in fns:
            self.data.append(h5py.File(fn, 'r'))
        datasets = ['"{}"'.format(hf.attrs['dataset']) for hf in self.data]
        if not eval('=='.join(datasets)):
            raise ValueError('[ERROR] All dataset should be same!')

        self.task_type = self.data[0].attrs['type']
        self.n_jobs = n_jobs

        # initiate pre-processor
        if preproc is not None:
            if preproc == 'standardize':
                self.preproc = StandardScaler()
            elif preproc == 'pca_whiten':
                self.preproc = PCA(n_components=256, whiten=True)
            else:
                raise ValueError(
                    '[ERROR] only supports "standardize" \
                    and "pca_whiten" at the moment!')
        else:
            self.preproc = FunctionTransformer(lambda x: x)

        self.model = None

    @staticmethod
    def _check_n_fix_data(X, y, report=False):
        """"""
        # check nan
        nan_samples = np.where(np.isnan(X).sum(axis=1) > 0)[0]
        normal_samples = np.where(np.isnan(X).sum(axis=1) == 0)[0]

        if report:
            print('Total {:d} NaN samples found'.format(len(nan_samples)))

        return X[normal_samples], y[normal_samples]

    def prepare_data(self):
        """"""
        # concatenate features
        X = np.concatenate(
            [
                np.concatenate(
                    [data['X'][t.decode()] for t in data.attrs['targets']],
                    axis=1
                )
                if data['X'].keys()[0] != 'fc'
                else data['X']['fc'][:]
                for data in self.data
            ], axis=1
        )

        y_true = self.data[0]['y'][:]
        X, y_true = self._check_n_fix_data(X, y_true, True)
        return X, y_true

    @abstractmethod
    def evaluate(self):
        pass


class MLEvaluator(BaseExternalTaskEvaluator):
    """"""
    def __init__(self, fns, preproc=None, n_jobs=-1):
        """"""
        super(MLEvaluator, self).__init__(fns, preproc, n_jobs)

        self.n_cv = 4
        self.tune_params = [
            {'kernel': ['rbf'], 'C': [0.001, 0.1, 1, 10, 100],
             'gamma': [1e-3, 1e-4]},
            {'kernel': ['linear'], 'C': [0.001, 0.1, 1, 10, 100]}]
        if self.task_type == 'classification':
            # self.model = SVC
            # self.model = GridSearchCV(SVC(), self.tune_params, cv=self.n_cv,
            #                           n_jobs=n_jobs)
            self.model = SVC(kernel='linear')
        elif self.task_type == 'regression':
            # self.model = SVR
            # self.model = GridSearchCV(SVR(), self.tune_params, cv=self.n_cv,
            #                           n_jobs=n_jobs)
            self.model = SVR()
        else:
            raise ValueError(
                '[ERROR] MLEvaluator only suport classification\
                or regression!')

        # setup pipeline
        self.pipeline = Pipeline(
            steps=[('sclr', self.preproc), ('model', self.model)])

    def evaluate(self):
        """"""
        X, y_true = self.prepare_data()
        print(X.shape)
        if y_true.ndim == 2 and y_true.shape[-1] == 1:
            y_true = y_true.ravel()

        t = time.time()
        cv = StratifiedKFold(10, shuffle=True)
        y_pred = cross_val_predict(self.pipeline, X, y_true, cv=cv)
        # y_t, y_p = [], []
        # for train_idx, test_idx in cv.split(X, y_true):
        #     # get split
        #     X_train, X_test = X[train_idx], X[test_idx]
        #     y_train, y_test = y_true[train_idx], y_true[test_idx]
        #     # re-init model
        #     mdl = GridSearchCV(self.model(), self.tune_params, cv=self.n_cv,
        #                        n_jobs=1, verbose=1)
        #     self.pipeline = Pipeline(
        #         steps=[('sclr', self.preproc), ('model', mdl)])
        #     # fit with hyper-param search
        #     self.pipeline.fit(X_train, y_train)
        #     print(self.pipeline.named_steps['model'].best_params_)
        #     # save results
        #     y_t.append(y_test)
        #     y_p.append(self.pipeline.predict(X_test))
        # y_true = np.concatenate(y_t, axis=0)
        # y_pred = np.concatenate(y_p, axis=0)
        cv_time = time.time() - t

        if self.task_type == 'classification':
            labels = self.data[0]['labels'][:]
            y_pred = y_pred.astype(int)
            y_true = y_true.astype(int)

            cm = confusion_matrix(
                labels[y_true], labels[y_pred], labels=labels)
            cr = classification_report(y_true, y_pred, target_names=labels)
            ac = accuracy_score(y_true, y_pred)

            return {'classification_report': cr,
                    'confusion_matrix': (cm, labels),
                    'accuracy_score': ac,
                    'time': cv_time}

        elif self.task_type == 'regression':
            labels, cm, cr = None, None, None
            ac = r2_score(y_true, y_pred)

            return {'classification_report': cr,
                    'confusion_matrix': (cm, labels),
                    'r2_score': ac,
                    'time': cv_time}


class RecSysEvaluator(BaseExternalTaskEvaluator):
    """"""
    def __init__(self, fns, preproc=None, n_jobs=-1, k=10, cv=10,
                 eval_type='outer', bias=False, n_factors=10, max_iter=20):
        """"""
        super(RecSysEvaluator, self).__init__(fns, preproc, n_jobs)

        self.eval_type = eval_type
        self.cv = cv
        self.k = k

        if self.task_type == 'recommendation':
            self.model = ContentExplicitALS(n_factors=n_factors, bias=bias,
                                            max_iter=max_iter, verbose=False)
            # self.model = ExplicitALS(n_factors=n_factors, bias=bias,
            #                          max_iter=max_iter, verbose=False)
        else:
            raise ValueError(
                '[ERROR] RecSysEvaluator only suports recommendation!')

    @staticmethod
    def score_at_k(model, k, R_test, X_test=None, verbose=False):
        """"""
        recall = []
        precision = []
        ap = []
        ndcg = []

        # prediction
        if X_test is None:
            R_pred = model.predict()
        else:
            R_pred = model.predict_from_side(X_test)

        if verbose:
            iterator = tqdm(zip(R_test, R_pred))
        else:
            iterator = zip(R_test, R_pred)

        for r, r_ in iterator:
            # true = np.where(r > 0)[0].astype(float).ravel()
            if np.sum(r) == 0:
                continue

            # relavance sort by pred rank
            rel = r[np.argsort(r_)][::-1]
            recall.append(np.sum(rel[:k]) / np.sum(r))
            precision.append(np.sum(rel[:k]) / k)
            ndcg.append(ndcg_at_k(rel, k, method=0))
            ap.append(apk(np.where(r)[0].tolist(),
                          np.argsort(r_)[::-1], k=k))

            # pred_at_k = np.argsort(r_)[-k:]
            # pred_rank_at_k = [
            #     np.where(pred_at_k == t)[0].tolist() for t in true]
            # hit_at_k = map(
            #     lambda x:x[0], filter(lambda x:len(x) > 0, pred_rank_at_k))
            # recall.append(float(len(hit_at_k)) / len(true))
            # precision.append(float(len(hit_at_k)) / k)

        # mean recall@k over users
        return np.mean(precision), np.mean(recall), np.mean(ndcg), np.mean(ap)

    @staticmethod
    def _make_cv(R, n_cv=5, typ='outer'):
        """"""
        Rs = []
        split_cv = []
        kf = KFold(n_splits=n_cv, shuffle=True)

        if typ == 'outer':
            triplet = _db2triplet(R)
            for trn_id, val_id in kf.split(range(R.shape[1])):
                split_cv.append((trn_id, val_id))
                trn_track_set = set(trn_id)
                val_track_set = set(val_id)
                train = np.array(filter(lambda x: x[1] in trn_track_set,
                                        triplet))
                valid = np.array(filter(lambda x: x[1] in val_track_set,
                                        triplet))
                Rs.append(
                    (
                        sp.coo_matrix(
                            (train[:, 2], (train[:, 0], train[:, 1])),
                            shape=R.shape).toarray(),
                        sp.coo_matrix(
                            (valid[:, 2], (valid[:, 0], valid[:, 1])),
                            shape=R.shape).toarray()
                    )
                )
            return Rs, split_cv

        elif typ == 'inner':
            triplet = _db2triplet(R)
            for trn_id, val_id in kf.split(triplet):
                train = triplet[trn_id].copy()
                valid = triplet[val_id].copy()
                split_cv.append(None)
                Rs.append(
                    (
                        sp.coo_matrix(
                            (train[:, 2], (train[:, 0], train[:, 1])),
                            shape=R.shape).toarray(),
                        sp.coo_matrix(
                            (valid[:, 2], (valid[:, 0], valid[:, 1])),
                            shape=R.shape).toarray()
                    )
                )
            return Rs, split_cv

    def _cross_val_score(self, R, X=None):
        """"""
        res = []

        # 1. make cv
        cv, split_cv = self._make_cv(R, n_cv=self.cv, typ=self.eval_type)

        # 2.1. for each cv, run predict
        # 2.2. and score them
        for (train, valid), split in zip(cv, split_cv):
            # train
            self.model.fit(R=train, X=X)

            # valid
            res.append(
                self.score_at_k(self.model, self.k, valid, X_test=X))

        # 3. return result
        return res

    def evaluate(self):
        """"""
        # concatenate features
        X, R = self.prepare_data()
        R = R.T  # (user, item)
        X = self.preproc.fit_transform(X)

        # shuffle dataset
        n, m = R.shape
        rnd_u = np.random.choice(n, n, replace=False)
        rnd_i = np.random.choice(m, m, replace=False)
        R = R[rnd_u][:, rnd_i]
        # X = None
        X = X[rnd_i]
        # X = np.random.rand(*X.shape)

        t = time.time()
        scores = self._cross_val_score(R=R, X=X)
        cv_time = time.time() - t

        precision = np.mean(map(lambda x: x[0], scores))
        recall = np.mean(map(lambda x: x[1], scores))
        ndcg = np.mean(map(lambda x: x[2], scores))
        ap = np.mean(map(lambda x: x[3], scores))

        return {'classification_report': None,
                'confusion_matrix': None,
                'recall@{:d}_score'.format(self.k): recall,
                'precision@{:d}_score'.format(self.k): precision,
                'ndcg@{:d}_score'.format(self.k): ndcg,
                'ap@{:d}_score'.format(self.k): ap,
                'time': cv_time}


def _make_hash(list_):
    return OrderedDict([(v, k) for k, v in enumerate(list_)])


def _make_sp_mat(triplet, track_hash, user_hash):
    return sp.coo_matrix(
        (
            len(triplet) * [1],
            (
                map(lambda x: user_hash[x[0]], triplet),
                map(lambda x: track_hash[x[1]], triplet)
            )
        ),
        dtype=int,
        shape=(len(user_hash.keys()), len(track_hash.keys()))
    )


def _split_outer(R, trn_idx, val_idx):
    """"""
    keep_track_train = trn_idx
    keep_track_valid = val_idx

    train = R[:, keep_track_train]
    valid = R[:, keep_track_valid]

    # print('train_user_fail', (train.sum(axis=1)==0).sum())
    # print('test_user_fail', (valid.sum(axis=1)==0).sum())
    # print('test_item_fail', (valid.sum(axis=0)==0).sum())

    keep_user_train = np.where(train.sum(axis=1) > 0)[0]
    keep_user_valid = list(
        set(np.where(valid.sum(axis=1) > 0)[0]) & set(keep_user_train)
    )

    return {
        'train': {'user': keep_user_train, 'track': keep_track_train},
        'valid': {'user': keep_user_valid, 'track': keep_track_valid}
    }


""" code from https://www.kaggle.com/wendykan/ndcg-example """


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def _db2triplet(R):
    """"""
    return np.vstack(sp.find(R)).T
