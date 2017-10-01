import time
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as sp

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (classification_report,
                             accuracy_score,
                             r2_score,
                             confusion_matrix)

from model.recsys.model import ContentExplicitALS, ContentImplicitALS
import h5py

class BaseExternalTaskEvaluator(object):
    __metaclass__ = ABCMeta

    """"""
    def __init__(self, fns, preproc=None, n_jobs=-1):
        """"""
        self.data = []
        for fn in fns:
            self.data.append(h5py.File(fn,'r'))
        if not eval('=='.join(
            ['"{}"'.format(hf.attrs['dataset']) for hf in self.data])):
            raise ValueError('[ERROR] All dataset should be same!')

        self.task_type = self.data[0].attrs['type']

        # initiate pre-processor
        if preproc is not None:
            if preproc == 'standardize':
                self.preproc = StandardScaler()
            elif preproc == 'pca_whiten':
                self.preproc = PCA(n_components=256, whiten=True)
        else:
            self.preproc = FunctionTransformer(lambda x:x)

        self.model = None

    @staticmethod
    def _check_n_fix_data(X, y, report=False):
        """"""
        # check nan
        nan_samples = np.where(np.isnan(X).sum(axis=1)>0)[0]
        normal_samples = np.where(np.isnan(X).sum(axis=1)==0)[0]

        if report:
            print('Total {:d} NaN samples found'.format(len(nan_samples)))

        return X[normal_samples], y[normal_samples]

    def prepare_data(self):
        """"""
        # concatenate features
        X = np.concatenate(
            [data['X'][:] for data in self.data], axis=1)

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

        if self.task_type == 'classification':
            self.model = SVC(kernel='linear')
        elif self.task_type == 'regression':
            self.model = SVR()
        else:
            raise ValueError(
                '[ERROR] MLEvaluator only suport classification or regression!')

        # setup pipeline
        self.pipeline = Pipeline(
            steps=[('sclr', self.preproc), ('model', self.model)])

    def evaluate(self):
        """"""
        X, y_true = self.prepare_data()
        if y_true.ndim == 2 and y_true.shape[-1] == 1:
            y_true = y_true.ravel()

        t = time.time()
        y_pred = cross_val_predict(self.pipeline, X, y_true, cv=10)
        cv_time = time.time() - t

        if self.task_type == 'classification':
            labels = self.data[0]['labels'][:]
            y_pred = y_pred.astype(int)
            y_true = y_true.astype(int)

            cm = confusion_matrix(labels[y_true], labels[y_pred], labels=labels)
            cr = classification_report(y_true, y_pred, target_names=labels)
            ac = accuracy_score(y_true, y_pred)

            return {'classification_report':cr, 'confusion_matrix':(cm, labels),
                    'accuracy_score':ac, 'time':cv_time}

        elif self.task_type == 'regression':
            labels, cm, cr = None, None, None
            ac = r2_score(y_true, y_pred)

            return {'classification_report':cr, 'confusion_matrix':(cm, labels),
                    'r2_score':ac, 'time':cv_time}



class RecSysEvaluator(BaseExternalTaskEvaluator):
    """"""
    def __init__(self, fns, preproc=None, n_jobs=-1, k=80, cv=5,
                 eval_type='outer', n_factors=40, max_iter=20):
        """"""
        super(RecSysEvaluator, self).__init__(fns, preproc, n_jobs)

        self.eval_type = eval_type
        self.cv = cv
        self.k = k

        if self.task_type == 'recommendation':
            self.model = ContentExplicitALS(n_factors=n_factors,
                                            max_iter=max_iter, verbose=True)
            # self.model = ContentImplicitALS(n_factors=n_factors,
            #                                 max_iter=max_iter,
            #                                 lam_u=10, lam_v=100, lam_w=10,
            #                                 verbose=True)
        else:
            raise ValueError(
                '[ERROR] RecSysEvaluator only suports recommendation!')

    @staticmethod
    def prec_recall_at_k(model, k, R_test, X_test=None, verbose=False):
        """"""
        recall = []
        precision = []

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
            true = np.where(r > 0)[0]
            if len(true) == 0:
                continue

            pred_at_k = np.argsort(r_)[-k:]
            pred_rank_at_k = [np.where(pred_at_k == t)[0].tolist() for t in true]
            hit_at_k = map(
                lambda x:x[0], filter(lambda x:len(x) > 0, pred_rank_at_k))

            recall.append(float(len(hit_at_k)) / len(true))
            precision.append(float(len(hit_at_k)) / k)

        # mean recall@k over users
        return np.mean(precision), np.mean(recall)

    @staticmethod
    def _make_cv(R, n_cv=5, typ='outer'):
        """"""
        R = R.T
        Rs = []
        split_cv = []
        kf = KFold(n_splits=n_cv, shuffle=True)

        if typ == 'outer':
            triplet = _db2triplet(R)
            for trn_id, val_id in kf.split(range(R.shape[1])):
                split_cv.append(None)

                trn_track_set = set(trn_id)
                val_track_set = set(val_id)

                train = np.array(filter(lambda x:x[1] in trn_track_set,
                                        triplet))
                valid = np.array(filter(lambda x:x[1] in val_track_set,
                                        triplet))

                Rs.append(
                    (
                        sp.coo_matrix((train[:,2], (train[:,0], train[:,1])),
                                      shape=R.shape).toarray(),
                        sp.coo_matrix((valid[:,2], (valid[:,0], valid[:,1])),
                                      shape=R.shape).toarray()
                    )
                )

                # splits = _split_outer(R, trn_id, val_id)
                # split_cv.append(splits)
                # Rs.append(
                #     (R[:, splits['train']['track']][splits['train']['user']],
                #      R[:, splits['valid']['track']][splits['valid']['user']])
                # )

            return Rs, split_cv

        elif typ == 'inner':
            triplet = _db2triplet(R)
            for trn_id, val_id in kf.split(triplet):
                train = triplet[trn_id].copy()
                valid = triplet[val_id].copy()
                split_cv.append(None)
                Rs.append(
                    (
                        sp.coo_matrix((train[:,2], (train[:,0], train[:,1])),
                                      shape=R.shape).toarray(),
                        sp.coo_matrix((valid[:,2], (valid[:,0], valid[:,1])),
                                      shape=R.shape).toarray()
                    )
                )
            return Rs, split_cv

    def _cross_val_score(self, R, X=None):
        """"""
        res = []
        # X = np.random.rand(*X.shape) # for test

        # 1. make cv
        cv, split_cv = self._make_cv(R, n_cv=self.cv, typ=self.eval_type)

        # 2.1. for each cv, run predict
        # 2.2. and score them
        for (train, valid), split in zip(cv, split_cv):
            if self.eval_type == 'outer':
                if split_cv is None or X is None:
                    raise ValueError(
                        '[ERROR] split or X should be passed!')

                # train
                self.model.fit(R=train, X=X)

                # valid
                res.append(
                    self.prec_recall_at_k(self.model, self.k, valid, X))

            elif self.eval_type == 'inner':
                # train
                self.model.fit(R=train, X=X)

                # valid
                res.append(
                    self.prec_recall_at_k(self.model, self.k, valid, X))

        # 3. return result
        return res

    def evaluate(self):
        """"""
        # concatenate features
        X, R = self.prepare_data()

        t = time.time()
        scores = self._cross_val_score(R=R, X=X)
        cv_time = time.time() - t

        precision = np.mean(map(lambda x:x[0], scores))
        recall = np.mean(map(lambda x:x[1], scores))

        return {'classification_report':None, 'confusion_matrix':None,
                'recall@{:d}_score'.format(self.k):recall,
                'precision@{:d}_score'.format(self.k):precision,
                'time':cv_time}

def _make_hash(list_):
	return OrderedDict([(v,k) for k, v in enumerate(list_)])

def _make_sp_mat(triplet, track_hash, user_hash):
	return sp.coo_matrix(
		(
			len(triplet) * [1],
			(
				map(lambda x:user_hash[x[0]], triplet),
				map(lambda x:track_hash[x[1]], triplet)
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
        'train':{'user':keep_user_train, 'track':keep_track_train},
        'valid':{'user':keep_user_valid, 'track':keep_track_valid}
    }

def _db2triplet(R):
    """"""
    return np.vstack(sp.find(R)).T

