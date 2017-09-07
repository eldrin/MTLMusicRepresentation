import os
import cPickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import namedtupled
import librosa
import h5py

from model.model import Model
from utils.misc import get_in_shape

import fire

class ExternalTaskEvaluator:
    """"""
    def __init__(self, fn):
        """"""
        self.data = h5py.File(fn,'r')
        self.model = LogisticRegression()

    def evaluate(self):
        """"""
        X = self.data['X'][:]
        y_true = self.data['y'][:]
        labels = self.data['labels'][:]

        y_pred = cross_val_predict(self.model, X, y_true, cv=10).astype(int)
        print(classification_report(y_true, y_pred, target_names=labels))


def external_eval(fn):
    evaluator = ExternalTaskEvaluator(fn)
    evaluator.evaluate()


# Evaluator for all internal tasks
class InternalTaskEvaluator:
    """"""
    def __init__(self, fn):
        """"""
        # load configuration for model
        model_state = joblib.load(fn)
        self.config = namedtupled.map(model_state['config'])

        # variable set up
        self.tasks = self.config.task
        self.sr = self.config.hyper_parameters.sample_rate
        self.hop_sz = hop_sz # in second
        self.in_shape = get_in_shape(self.config)

        # load valid id for each task
        split = namedtupled.reduce(self.config.paths.meta_data.splits)
        self.valid_ids = {
            k:joblib.load(v)['valid']
            for k,v in split.iteritems()
        }
        self.path_map = pkl.load(open(self.config.paths.path_map))

        # load model builder
        self.model = Model(config, feature_layer)


# Task specific evaluation helpers
def evaluate_tag(song_ids, model, path_map, config, top_k=20):
    """"""
    # variable set up
    tasks = self.config.task
    sr = self.config.hyper_parameters.sample_rate
    hop_sz = hop_sz # in second
    in_shape = get_in_shape(self.config)

    # load tag factor
    tag_factor_model = joblib.load(
        os.path.join(
            config.paths.meta_data.root,
            config.paths.meta_data.targets.tag
        )
    )

    V = tag_factor_model['tag_factors']
    U = tag_factor_model['item_factors']
    tids_hash = {
        tid:j for j,tid in enumerate(tag_factor_model['tids'])
    }

    # inference song level
    for song_id in song_ids:
        y = U[tids_hash[song_id]]
        o, c, f = _get_feature_and_prob(
            path_map[song_id], y, model, hop_sz, in_shape[-1])

        pred = o.dot(V)
        pred_tag_ix = np.argsort(pred)[-top_k:][::-1]
        # true_tag_ix = something
        # TODO: finish this up


def _get_feature_and_prob(song_fn, Y, model, hop_sz, sr, dur_sp):
    """"""
    y, sr = librosa.load(song_fn, sr=sr)

    if y.ndim==1:
        y = np.repeat(y[None,:], 2, axis=0)

    X = []
    for start in xrange(0, len(y), hop_sz * sr):
        slc = slice(start, start + dur_sp)
        x = y[:,slc][None,:,:]
        if x.shape[-1] < dur_sp:
            continue
        X.append(x)
    X = np.array(X)
    Y = np.repeat(Y[None,:], X.shape[0], axis=0)

    O = model.predict('tag', X).mean(axis=0)
    C = model.cost('tag', X, Y).mean(axis=0)
    F_raw = model.feature(X)
    F = np.concatenate([F_raw.mean(axis=0), F_raw.std(axis=0)])

    return O, C, F

if __name__ == "__main__":
    fire.Fire(external_eval)
