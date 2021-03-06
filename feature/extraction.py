import os
import traceback

import namedtupled
from functools import partial
from itertools import chain

import numpy as np
from scipy import stats

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import librosa

import h5py

from model.preproc.model import MelSpectrogramGPU
from model.model import Model
from utils.misc import get_layer, load_config

import fire
import tqdm

DATASET_INFO = {
    'GTZAN': {
        'info': '/mnt/bulk2/datasets/GTZAN/GTZAN.dataset.info',
        'type': 'classification'
    },
    'Ballroom': {
        'info': '/mnt/bulk2/datasets/Ballroom/Ballroom.dataset.info',
        'type': 'classification'
    },
    # 'BallroomExt': {
    #     'info': '/mnt/bulk2/datasets/BallroomExt/BallroomExt.dataset.info',
    #     'type': 'classification'
    # },
    # 'FMA': {
    #     'info': '/mnt/bulk2/datasets/FMA/FMA_MEDIUM.dataset.info',
    #     'type': 'classification'
    # },
    'FMA_SUB': {
        'info': '/mnt/bulk2/datasets/FMA/FMA_MEDIUM_SUB.dataset.info',
        'type': 'classification'
    },
    'EmoValStatic': {
        'info': '/mnt/bulk2/datasets/MusicEmotion/MusicEmotionStaticValence.dataset.info',
        'type': 'regression'
    },
    'EmoAroStatic': {
        'info': '/mnt/bulk2/datasets/MusicEmotion/MusicEmotionStaticArousal.dataset.info',
        'type': 'regression'
    },
    'IRMAS_SUB': {
        'info': '/mnt/bulk2/datasets/IRMAS/IRMAS_SUB.dataset.info',
        'type': 'classification'
    },
    'ThisIsMyJam': {
        'info': '/mnt/bulk2/datasets/JamDataset/ThisIsMyJam.dataset.info',
        'type': 'recommendation'
    }
}


class BaseExtractor(object):
    """"""
    def __init__(self, task, out_dir=None, hop_sz=1.,
                 prob=False, *args, **kwargs):
        """"""
        if task not in DATASET_INFO:
            raise ValueError(
                '[ERROR] {} is not supported!'.format(task))
        self.task = task
        self.task_type = DATASET_INFO[self.task]['type']

        self.hop_sz = hop_sz  # in second
        self.prob = prob  # probability output

        if out_dir is None:
            self.root = os.getcwd()
        else:
            if os.path.exists(out_dir):
                self.root = out_dir
            else:
                raise ValueError(
                    '[ERROR] {} is not existing!'.format(self.root))

        # load db information data
        self.db_info = map(
            lambda r: (r[0], r[1], r[2].split(',')),
            map(lambda l: l.replace('\n', '').split('\t'),
                open(DATASET_INFO[task]['info'], 'r').readlines())
        )

        # task-specific symbolic target dimension
        if self.task_type != 'recommendation':
            l = 1
        else:
            l = max(list(chain.from_iterable(
                map(lambda r: [int(d) for d in r[2]], self.db_info))))
            l += 1
        self.label_dim = l

    def post_init(self):
        """"""
        # setup label dataset
        if self.task_type == 'classification':
            label_set = map(lambda x: x[-1], self.db_info)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(np.array(label_set).ravel())

            self.hf.create_dataset(
                'labels',
                data=np.array(self.label_encoder.classes_, dtype='S')
            )

        self.hf.attrs['dataset'] = self.task
        self.hf.attrs['type'] = DATASET_INFO[self.task]['type']

    def _prepare_db(self):
        """"""
        raise NotImplementedError

    def _extract_feature(self, fn):
        """"""
        raise NotImplementedError

    def _extract_label(self, label):
        """"""
        if self.task_type == 'classification':
            return self.label_encoder.transform(np.array(label).ravel())[0]
        elif self.task_type == 'regression':
            return float(label[0])
        elif self.task_type == 'recommendation':
            y = np.zeros((self.label_dim,))
            y[[int(d) for d in label]] = 1
            return y

    def _save_X(self, ix, fn):
        """"""
        raise NotImplementedError

    def _save_y(self, ix, label):
        """"""
        self.hf['y'][ix] = self._extract_label(label)

    def process(self):
        """"""
        for (ix, fn, label) in tqdm.tqdm(self.db_info, ncols=80):
            ix = int(ix)
            self._save_X(ix, fn)
            self._save_y(ix, label)


class MTLExtractor(BaseExtractor):
    """"""
    def __init__(self, model_fn, task, out_dir=None, hop_sz=1.):
        """"""
        super(MTLExtractor, self).__init__(task, out_dir, hop_sz, prob=True)

        # load configuration for model
        if os.path.exists(model_fn):
            model_id = os.path.splitext(os.path.basename(model_fn))[0]
            self.model_id = model_id.split('_state')[0]
            model_state = joblib.load(model_fn)
            self.config = namedtupled.map(model_state['config'])
        else:
            self.model_id = 'rnd'
            # load default config and change task as rand
            self.config = load_config('config/config.example.json')
            self.config.target[0] = 'rand'

        self.out_fn = os.path.join(
            self.root, self.model_id + '_{}_feature.h5'.format(self.task))

        self.targets = self.config.target

        # load model
        self.model = Model(self.config)

        # variable set up
        self.sr = self.config.hyper_parameters.sample_rate
        self.length = self.config.hyper_parameters.patch_length
        self.n_fft = self.config.hyper_parameters.n_fft
        self.hop_sz_trn = self.config.hyper_parameters.hop_size

        self.input = self.config.hyper_parameters.input
        self.hop = int(self.hop_sz * self.sr)
        sig_len = int(self.sr * self.length)
        self.sig_len = sig_len - sig_len % self.hop_sz_trn

        # prepare preprocessor if needed
        if self.config.hyper_parameters.input == 'melspec':
            self.melspec = MelSpectrogramGPU(
                2, self.sr, self.n_fft, self.hop_sz_trn)

        # set feature layer names
        branch_at = self.config.hyper_parameters.branch_at
        if isinstance(branch_at, (int, float)):
            self.feature_layers = [
                '{}.fc'.format(t)
                for t in self.targets
            ]

        elif isinstance(branch_at, (str, unicode)) and branch_at == "fc":
            self.feature_layers = ['fc']

        self._prepare_db()
        super(MTLExtractor, self).post_init()
        self.hf.attrs['targets'] = [t.encode() for t in self.targets]

    def _prepare_db(self):
        """ prepare task specific db setting """

        n = len(self.db_info)  # num obs
        # currently, we use same dim for all multi targets
        m = get_layer(
            self.model.net,
            self.feature_layers[0]).output_shape[-1]  # feature dim
        o = self.config.hyper_parameters.n_out  # model output dim

        self.hf = h5py.File(self.out_fn, 'w')
        self.hf.create_dataset('y', shape=(n, self.label_dim))

        self.hf.create_group('X')
        # 'FC' branch case, only one dataset needed
        if self.feature_layers[0] == 'fc':
            self.hf['X'].create_dataset('fc', shape=(n, m * 2))
        # otherwise, dataset needed per each task
        else:
            for target, n_out in zip(self.targets, o):
                # mean / std
                self.hf['X'].create_dataset(target, shape=(n, m * 2))

        if self.prob:
            self.hf.create_group('Z')
            for target, n_out in zip(self.targets, o):
                if target == 'self':
                    continue
                self.hf['Z'].create_dataset(target, shape=(n, n_out))

    def _extract_feature(self, fn):
        """"""
        # load audio
        y, _ = librosa.load(fn, sr=self.sr, res_type='kaiser_fast')
        if y.ndim < 2:
            y = np.repeat(y[None, :], 2, axis=0)

        end = y.shape[1]
        X = []
        feature = {target: None for target in self.targets}
        mean_prob = {target: [] for target in self.targets}
        for j in xrange(0, end, self.hop):
            slc = slice(j, j + self.sig_len)
            x_chunk = y[:, slc][None, :, :]
            if x_chunk.shape[2] < self.sig_len:
                continue
            if self.config.hyper_parameters.input == 'melspec':
                x_chunk = self.melspec.process(x_chunk)
            X.append(x_chunk)
        x = np.concatenate(X, axis=0)

        # 'FC' branching case, all feature are same
        if self.feature_layers[0] == 'fc':
            Y = self.model.feature(self.targets[0], x)
            feature['fc'] = np.concatenate(
                [np.mean(Y, axis=0).ravel(),
                 np.std(Y, axis=0).ravel()]).ravel()
        # other branching cases, need to extract each feature
        else:
            for target in self.targets:
                Y = self.model.feature(target, x)
                feature[target] = np.concatenate(
                    [np.mean(Y, axis=0).ravel(),
                     np.std(Y, axis=0).ravel()]).ravel()

        for target in self.targets:
            if target == 'self':
                continue
            mean_prob[target].append(
                self.model.predict(target, x).mean(axis=0).ravel())

        return feature, mean_prob

    def _save_X(self, ix, fn):
        """"""
        for target in self.targets:
            try:
                feat, mean_prob = self._extract_feature(fn)
                if self.feature_layers[0] == 'fc':
                    self.hf['X']['fc'][ix] = feat['fc']
                else:
                    self.hf['X'][target][ix] = feat[target]
                if target == 'self':
                    continue

                if self.prob:
                    self.hf['Z'][target][ix] = mean_prob[target]

            except Exception:
                traceback.print_exc()
                self.hf['X'][target][ix, :] = np.nan
                print('[ERROR] file {} has problem!'.format(fn))


class MFCCExtractor(BaseExtractor):
    """"""
    def __init__(self, task, out_dir=None, hop_sz=1., n_mfcc=20):
        """"""
        super(MFCCExtractor, self).__init__(task, out_dir, hop_sz)

        self.out_fn = os.path.join(
            self.root, 'mfcc_{}_feature.h5'.format(self.task))
        self.n_mfcc = n_mfcc
        self.prob = False
        self._prepare_db()

        super(MFCCExtractor, self).post_init()
        self.hf.attrs['targets'] = ['mfcc']

    def _prepare_db(self):
        """"""
        n = len(self.db_info)  # num obs
        # m = self.n_mfcc * 3 * 2  # mfcc x (mfcc + d + dd) x (avg + std)
        m = self.n_mfcc * 7  # mfcc * (avg, std, skew, kurt, median, min, max)

        self.hf = h5py.File(self.out_fn, 'w')
        self.hf.create_dataset('y', shape=(n, self.label_dim))

        self.hf.create_group('X')
        self.hf['X'].create_dataset('mfcc', shape=(n, m))

    def _extract_feature(self, fn):
        """"""
        y, sr = librosa.load(fn, sr=22050, mono=True)
        M = librosa.feature.mfcc(y, sr, n_mfcc=self.n_mfcc)
        # dM = M[:, 1:] - M[:, :-1]
        # ddM = dM[:, 1:] - dM[:, :-1]

        # X = list(chain.from_iterable(
        #     map(lambda x:
        #         (x.mean(axis=1), x.std(axis=1)),
        #         [M, dM, ddM])
        # ))

        X = [
            np.mean(M, axis=1),
            np.std(M, axis=1),
            stats.skew(M, axis=1),
            stats.kurtosis(M, axis=1),
            np.median(M, axis=1),
            np.min(M, axis=1),
            np.max(M, axis=1)
        ]
        return np.concatenate(X)

    def _save_X(self, ix, fn):
        """"""
        self.hf['X']['mfcc'][ix] = self._extract_feature(fn)


def main(task, feature, out_dir, hop_sz=1.):
    """
    feature : {MTL_model_fn or 'mfcc'}
    """

    if feature == 'mfcc':
        Extractor = MFCCExtractor
    else:
        Extractor = partial(MTLExtractor, model_fn=feature)

    if task.lower() == 'all':  # do it all
        for task_, info in DATASET_INFO.iteritems():
            ext = Extractor(
                task=task_,
                hop_sz=hop_sz,
                out_dir=out_dir
            )
            ext.process()
    else:  # individual tasks
        ext = Extractor(
            task=task,
            hop_sz=hop_sz,
            out_dir=out_dir
        )
        ext.process()


if __name__ == "__main__":
    fire.Fire(main)
