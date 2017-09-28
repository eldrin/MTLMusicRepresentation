import os
import traceback

import namedtupled
from itertools import izip_longest

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import librosa

import h5py

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
    'BallroomExt':{
        'info':'/mnt/bulk2/datasets/BallroomExt/BallroomExt.dataset.info',
        'type':'classification'
    },
    'FMA':{
        'info':'/mnt/bulk2/datasets/FMA/FMA_MEDIUM.dataset.info',
        'type':'classification'
    },
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
    }
}

class FeatureExtractor:
    """"""
    def __init__(
        self, fn, task, feature_layer, out_dir=None,
        cam_layer='conv5.1', hop_sz=1., *args, **kwargs):
        """"""
        if task not in DATASET_INFO:
            raise ValueError(
                '[ERROR] {} is not supported!'.format(task))
        self.task = task

        # load configuration for model
        model_state = joblib.load(fn)

        self.config = namedtupled.map(model_state['config'])
        self.db_info = map(
            lambda l:l.replace('\n','').split('\t'),
            open(DATASET_INFO[task]['info'],'r').readlines())

        self.targets = self.config.target

        # variable set up
        self.sr = self.config.hyper_parameters.sample_rate
        self.hop_sz = hop_sz # in second
        self.in_shape = get_in_shape(self.config)

        # load model builder
        self.model = Model(self.config, feature_layer)

        if DATASET_INFO[task]['type'] == 'classification':
            # process label set to one-hot coding
            label_set = map(lambda x:x[-1],self.db_info)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(label_set)

        # open and initiate dump file (hdf5)
        model_id = os.path.splitext(os.path.basename(fn))[0]
        model_id = model_id.split('_state')[0]
        if out_dir is None:
            root = os.getcwd()
        else:
            if os.path.exists(out_dir):
                root = out_dir
            else:
                raise ValueError(
                    '[ERROR] {} is not existing!')

        out_fn = os.path.join(
            root, model_id + '_{}_feature.h5'.format(task))

        n = len(self.db_info) # num obs
        m = get_layer(
            self.model.net, feature_layer).output_shape[-1] # feature dim
        o = self.config.hyper_parameters.n_out # model output dim

        self.hf = h5py.File(out_fn,'w')
        self.hf.create_dataset('X', shape=(n, m * 2)) # mean / std
        self.hf.create_dataset('y', shape=(n,))
        self.hf.create_group('Z')
        for target, n_out in zip(self.targets, o):
            if target == 'self': continue
            self.hf['Z'].create_dataset(target, shape=(n, n_out))

        if DATASET_INFO[task]['type'] == 'classification':
            self.hf.create_dataset(
                'labels',
                data=np.array(self.label_encoder.classes_, dtype='S')
            )

        self.hf.attrs['dataset'] = task
        self.hf.attrs['type'] = DATASET_INFO[task]['type']

    def process(self):
        """"""
        m = self.hf['X'].shape[-1]/2
        hop_samples = int(self.hop_sz * self.sr)

        for (ix, fn, label) in tqdm.tqdm(self.db_info):
            ix = int(ix)
            try:
                # load audio
                y, _ = librosa.load(fn, sr=self.sr, res_type='kaiser_fast')
                if y.ndim < 2:
                    y = np.repeat(y[None,:],2,axis=0)

                end = y.shape[-1]
                X = []
                Z = {target:[] for target in self.targets}
                for j in xrange(0, end, hop_samples):
                    slc = slice(j, j + self.in_shape[-1])
                    x_chunk = y[:,slc][None,:,:]

                    if x_chunk.shape[-1] < self.in_shape[-1]:
                        continue

                    X.append(self.model.feature(x_chunk))

                    for target in self.targets:
                        if target == 'self': continue
                        Z[target].append(self.model.predict(target, x_chunk))

                self.hf['X'][ix, :m] = np.mean(X, axis=0)
                self.hf['X'][ix, m:] = np.std(X, axis=0)
                for target in self.targets:
                    if target == 'self': continue
                    self.hf['Z'][target][ix] = np.mean(Z[target], axis=0)

            except Exception as e:
                traceback.print_exc()
                self.hf['X'][ix,:] = np.nan
                print('[ERROR] file {} has problem!'.format(fn))

            finally:
                if DATASET_INFO[self.task]['type'] == 'classification':
                    self.hf['y'][ix] = self.label_encoder.transform([label])[0]
                elif DATASET_INFO[self.task]['type'] == 'regression':
                    self.hf['y'][ix] = float(label)


def main(task, model_state_fn, out_dir, hop_sz=1., feature_layer='fc'):
    """"""
    ext = FeatureExtractor(
        fn=model_state_fn,
        task=task,
        hop_sz=hop_sz,
        feature_layer=feature_layer,
        out_dir=out_dir
    )
    ext.process()


if __name__ == "__main__":
    fire.Fire(main)
