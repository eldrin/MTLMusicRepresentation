import time

import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (classification_report,
                             accuracy_score,
                             r2_score,
                             confusion_matrix)

import h5py

class ExternalTaskEvaluator:
    """"""
    def __init__(self, fns, preproc=None, n_jobs=-1):
        """"""
        self.data = []
        for fn in fns:
            self.data.append(h5py.File(fn,'r'))
        if not eval('=='.join(
            ['"{}"'.format(hf.attrs['dataset']) for hf in self.data])):
            raise ValueError('[ERROR] All dataset should be same!')

        # initiate evaluation model
        if self.data[0].attrs['type'] == 'classification':
            # self.model = LogisticRegression(
            #     solver='saga', max_iter=100, n_jobs=n_jobs,
            #     multi_class='multinomial'
            #     # multi_class='ovr'
            # )
            self.model = SVC(kernel='linear')
        elif self.data[0].attrs['type'] == 'regression':
            # self.model = LinearRegression(n_jobs=n_jobs)
            self.model = SVR()

        # initiate pre-processor
        if preproc is not None:
            if preproc == 'standardize':
                self.preproc = StandardScaler()
            elif preproc == 'pca_whiten':
                self.preproc = PCA(n_components=256, whiten=True)
        else:
            self.preproc = FunctionTransformer(lambda x:x)

        # setup pipeline
        self.pipeline = Pipeline(
            steps=[('sclr', self.preproc), ('classifier', self.model)])

    def evaluate(self):
        """"""
        # concatenate features
        X = np.concatenate(
            [data['X'][:] for data in self.data], axis=1)

        y_true = self.data[0]['y'][:]
        X, y_true = self._check_n_fix_data(X, y_true, True)

        t = time.time()
        y_pred = cross_val_predict(self.pipeline, X, y_true, cv=10)
        cv_time = time.time() - t

        if self.data[0].attrs['type'] == 'classification':
            labels = self.data[0]['labels'][:]
            y_pred = y_pred.astype(int)
            y_true = y_true.astype(int)

            cm = confusion_matrix(labels[y_true], labels[y_pred], labels=labels)
            cr = classification_report(y_true, y_pred, target_names=labels)
            ac = accuracy_score(y_true, y_pred)
        elif self.data[0].attrs['type'] == 'regression':
            labels, cm, cr = None, None, None
            ac = r2_score(y_true, y_pred)

        return {'classification_report':cr, 'confusion_matrix':(cm, labels),
                'accuracy':ac, 'time':cv_time}

    @staticmethod
    def _check_n_fix_data(X, y, report=False):
        """"""
        # check nan
        nan_samples = np.where(np.isnan(X).sum(axis=1)>0)[0]
        normal_samples = np.where(np.isnan(X).sum(axis=1)==0)[0]

        if report:
            print('Total {:d} NaN samples found'.format(len(nan_samples)))

        return X[normal_samples], y[normal_samples]
