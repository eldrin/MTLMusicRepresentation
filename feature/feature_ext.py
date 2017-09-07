import os

import namedtupled

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import librosa

import h5py

from model.model import Model
from model.helper import load_check_point, get_debug_funcs
from utils.misc import get_in_shape, get_layer

import fire
import tqdm

DATASET_FNS = {
    'GTZAN':'/mnt/bulk2/datasets/GTZAN/GTZAN.dataset.info',
    'Ballroom':'/mnt/bulk2/datasets/Ballroom/Ballroom.dataset.info',
    'FMA':'/mnt/bulk2/datasets/FMA/FMA_MEDIUM.dataset.info'
}

class FeatureExtractor:
    """"""
    def __init__(
        self, target_dataset, model_root, model_id, feature_layer,
        cam_layer='conv5.1', hop_sz=1., *args, **kwargs):
        """"""
        if target_dataset not in DATASET_FNS:
            raise ValueError(
                '[ERROR] {} is not supported!'.format(target_dataset))

        # load configuration for model
        model_state_fn = os.path.join(
            model_root, model_id + '_state.dat.gz')
        model_state = joblib.load(model_state_fn)

        self.config = namedtupled.map(model_state['config'])
        self.db_info = map(
            lambda l:l.replace('\n','').split('\t'),
            open(DATASET_FNS[target_dataset],'r').readlines())

        self.tasks = self.config.task

        # variable set up
        self.sr = self.config.hyper_parameters.sample_rate
        self.hop_sz = hop_sz # in second
        self.in_shape = get_in_shape(self.config)

        # load model builder
        self.model = Model(config, feature_layer)

        # process label set to one-hot coding
        label_set = map(lambda x:x[-1],self.db_info)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(label_set)

        # open and initiate dump file (hdf5)
        out_fn = os.path.join(
            os.getcwd(), model_id + '_feature.h5')

        n = len(self.db_info) # num obs
        m = get_layer(self.net['out'], feature_layer).output_shape[-1] # feature dim
        l = len(self.label_encoder.classes_) # dataset label
        o = self.config.hyper_parameters.n_out # model output dim

        self.hf = h5py.File(out_fn,'w')
        self.hf.create_dataset('X', shape=(n, m*2)) # mean / std
        self.hf.create_dataset('y', shape=(n,))
        self.hf.create_group('Z')
        for task, n_out in zip(self.tasks, o):
            self.hf['Z'].create_dataset(task, shape=(n, n_out))
        self.hf.create_dataset(
            'labels',
            data=np.array(self.label_encoder.classes_, dtype='S')
        )

    def process(self):
        """"""
        m = self.hf['X'].shape[-1]/2
        hop_samples = int(self.hop_sz * self.sr)

        for (ix, fn, label) in tqdm.tqdm(self.db_info):

            # load audio
            y, _ = librosa.load(fn, sr=self.sr, res_type='kaiser_fast')
            if y.ndim < 2:
                y = np.repeat(y[None,:],2,axis=0)

            end = y.shape[-1]
            X = []
            Z = {task:[] for task in self.tasks}
            for j in xrange(0, end, hop_samples):
                slc = slice(j, j + self.in_shape[-1])
                x_chunk = y[:,slc][None,:,:]

                if x_chunk.shape[-1] < self.in_shape[-1]:
                    continue

                X.append(self.model.features(x_chunk))

                for task in self.tasks:
                    Z[task].append(self.model.predict(x_chunk))

            ix = int(ix)
            self.hf['X'][ix, :m] = np.mean(X, axis=0)
            self.hf['X'][ix, m:] = np.std(X, axis=0)
            for task in self.tasks:
                self.hf['Z'][task][ix] = np.mean(Z[task], axis=0)
            self.hf['y'][ix] = self.label_encoder.transform([label])[0]


def main(target, model_root, model_id, hop_sz=1., feature_layer='fc.bn.do'):
    """"""
    ext = FeatureExtractor(
        target_dataset=target,
        model_root=model_root,
        model_id=model_id,
        hop_sz=hop_sz,
        faeture_layer=feature_layer
    )
    ext.process()


if __name__ == "__main__":
    fire.Fire(main)
