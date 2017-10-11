import os
import traceback

import namedtupled
from itertools import izip_longest, chain

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import librosa

import h5py

from model.preproc.model import MelSpectrogramGPU
from model.model import Model
from model.helper import load_check_point, get_debug_funcs
from utils.misc import get_in_shape, get_layer, pmap, load_audio

import fire
import tqdm

DATASET_INFO = {
    'GTZAN':{
        'info':'/mnt/bulk2/datasets/GTZAN/GTZAN.dataset.info',
        'type':'classification'
    },
    'Ballroom':{
        'info':'/mnt/bulk2/datasets/Ballroom/Ballroom.dataset.info',
        'type':'classification'
    },
    # 'BallroomExt':{
    #     'info':'/mnt/bulk2/datasets/BallroomExt/BallroomExt.dataset.info',
    #     'type':'classification'
    # },
    # 'FMA':{
    #     'info':'/mnt/bulk2/datasets/FMA/FMA_MEDIUM.dataset.info',
    #     'type':'classification'
    # },
    'FMA_SUB':{
        'info':'/mnt/bulk2/datasets/FMA/FMA_MEDIUM_SUB.dataset.info',
        'type':'classification'
    },
    'EmoValStatic':{
        'info':'/mnt/bulk2/datasets/MusicEmotion/MusicEmotionStaticValence.dataset.info',
        'type':'regression'
    },
    'EmoAroStatic':{
        'info':'/mnt/bulk2/datasets/MusicEmotion/MusicEmotionStaticArousal.dataset.info',
        'type':'regression'
    },
    'IRMAS_SUB':{
        'info':'/mnt/bulk2/datasets/IRMAS/IRMAS_SUB.dataset.info',
        'type':'classification'
    },
    'ThisIsMyJam':{
        'info':'/mnt/bulk2/datasets/JamDataset/ThisIsMyJam.dataset.info',
        'type':'recommendation'
    }
}

class BaseExtractor(object):
    """"""
    def __init__(
        self, fn, task, feature_layer, out_dir=None,
        cam_layer='conv5.1', hop_sz=1., *args, **kwargs):
        """"""
        if task not in DATASET_INFO:
            raise ValueError(
                '[ERROR] {} is not supported!'.format(task))
        self.task = task
        self.task_type = DATASET_INFO[self.task]['type']

        model_id = os.path.splitext(os.path.basename(fn))[0]
        self.model_id = model_id.split('_state')[0]
        if out_dir is None:
            self.root = os.getcwd()
        else:
            if os.path.exists(out_dir):
                self.root = out_dir
            else:
                raise ValueError(
                    '[ERROR] {} is not existing!')

        # load configuration for model
        model_state = joblib.load(fn)
        self.config = namedtupled.map(model_state['config'])

        # load db information data
        self.db_info = map(
            lambda r: (r[0], r[1], r[2].split(',')),
            map(lambda l:l.replace('\n','').split('\t'),
            open(DATASET_INFO[task]['info'],'r').readlines())
        )
        self.targets = self.config.target

        # load model
        self.model = Model(self.config, feature_layer)

        # variable set up
        self.sr = self.config.hyper_parameters.sample_rate
        self.length = self.config.hyper_parameters.patch_length
        self.n_fft = self.config.hyper_parameters.n_fft
        self.hop_sz_trn = self.config.hyper_parameters.hop_size
        self.hop_sz = hop_sz # in second

        self.input = self.config.hyper_parameters.input
        self.hop = int(self.hop_sz * self.sr)
        sig_len = int(self.sr * self.length)
        self.sig_len = sig_len - sig_len % self.hop_sz_trn

        # prepare preprocessor if needed
        if self.config.hyper_parameters.input == 'melspec':
            self.melspec = MelSpectrogramGPU(
                2, self.sr, self.n_fft, self.hop_sz_trn)

        self.feature_layer = feature_layer

        # prepare database
        self.prepare_db()

    def prepare_db(self):
        """ prepare task specific db setting """

        # open and initiate dump file (hdf5)
        out_fn = os.path.join(
            self.root, self.model_id + '_{}_feature.h5'.format(self.task))

        # task-specific symbolic target dimension
        if self.task_type != 'recommendation':
            l = 1
            if self.task_type == 'classification':
                # label_dtype = 'S10'
                label_dtype = int
            elif self.task_type == 'regression':
                label_dtype = np.float32
        else:
            l = max(list(chain.from_iterable(
                map(lambda r:[int(d) for d in r[2]], self.db_info))))
            l += 1
            label_dtype = np.int32

        self.label_dim = l
        n = len(self.db_info) # num obs
        m = get_layer(
            self.model.net, self.feature_layer).output_shape[-1] # feature dim
        o = self.config.hyper_parameters.n_out # model output dim

        self.hf = h5py.File(out_fn,'w')
        self.hf.create_dataset('X', shape=(n, m * 2)) # mean / std
        self.hf.create_dataset('y', shape=(n, l))

        self.hf.create_group('Z')
        for target, n_out in zip(self.targets, o):
            if target == 'self': continue
            self.hf['Z'].create_dataset(target, shape=(n, n_out))

        if self.task_type == 'classification':
            label_set = map(lambda x:x[-1], self.db_info)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(np.array(label_set).ravel())

            self.hf.create_dataset(
                'labels',
                data=np.array(self.label_encoder.classes_, dtype='S')
            )

        self.hf.attrs['dataset'] = self.task
        self.hf.attrs['type'] = DATASET_INFO[self.task]['type']

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

    def _extract_feature(self, fn):
        """"""
        # load audio
        y, _ = librosa.load(fn, sr=self.sr, res_type='kaiser_fast')
        if y.ndim < 2:
            y = np.repeat(y[None,:],2,axis=0)

        end = y.shape[1]
        X = {target:[] for target in self.targets}
        mean_prob = {target:[] for target in self.targets}
        for j in xrange(0, end, self.hop):
            slc = slice(j, j + self.sig_len)

            x_chunk = y[:,slc][None,:,:]

            if x_chunk.shape[2] < self.sig_len:
                continue

            if self.config.hyper_parameters.input == 'melspec':
                x_chunk = self.melspec.process(x_chunk)

            for target in self.targets:
                if target == 'self': continue
                X[target].append(self.model.feature(target, x_chunk))
                mean_prob[target].append(
                    self.model.predict(target, x_chunk))

        if len(self.targets) > 1:
            raise NotImplementedError(
                '[ERROR] multi target extraction is under construction!')
        else:
            target = self.targets[0]
            feature = np.concatenate(
                [np.mean(X[target], axis=0).ravel(), np.std(X[target], axis=0).ravel()])

        return feature, mean_prob

    def process(self):
        """"""
        for (ix, fn, label) in tqdm.tqdm(self.db_info, ncols=80):
            ix = int(ix)
            try:
                self.hf['X'][ix], mean_prob = self._extract_feature(fn)

                for target in self.targets:
                    if target == 'self': continue
                    self.hf['Z'][target][ix] = np.mean(
                        mean_prob[target], axis=0).ravel()

            except Exception as e:
                traceback.print_exc()
                self.hf['X'][ix,:] = np.nan
                print('[ERROR] file {} has problem!'.format(fn))
            finally:
                self.hf['y'][ix] = self._extract_label(label)


def main(task, model_state_fn, out_dir, hop_sz=1., feature_layer='fc'):
    """"""
    if task.lower() == 'all': # do it all
        for task_, info in DATASET_INFO.iteritems():
	    ext = BaseExtractor(
		fn=model_state_fn,
		task=task_,
		hop_sz=hop_sz,
		feature_layer=feature_layer,
		out_dir=out_dir
	    )
	    ext.process()
    else: # individual tasks
	ext = BaseExtractor(
	    fn=model_state_fn,
	    task=task,
	    hop_sz=hop_sz,
	    feature_layer=feature_layer,
	    out_dir=out_dir
        )
        ext.process()

if __name__ == "__main__":
    fire.Fire(main)
