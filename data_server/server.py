import os
import copy
from itertools import chain
import cPickle as pkl
import traceback

import numpy as np

from fuel.schemes import ShuffledScheme
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.server import start_server

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from functools import partial

import librosa
import fire

from utils.misc import pmap, load_audio_batch, load_config
from utils.misc import zero_pad_signals, load_audio, prepare_sub_batches
from helper import sample_matcher_idx

class MSD(IndexableDataset):
    """Assuming input datastream is a example of
    user-item interaction triplet. In this class
    mel-spectrogram and tag vector (BOW) is fetched
    based on the triplet's item index
    """
    provides_sources = ('raw')

    def __init__(
        self, target, which_set, config, *args, **kwargs):
        """
        """
        self.source = 'raw'
        self.axis_labels = None

        self.sr = config.hyper_parameters.sample_rate
        self.length = config.hyper_parameters.patch_length
        self.len_samples = int(self.length * self.sr)
        self.sub_batch_sz = config.hyper_parameters.sub_batch_size

        self.n_fft = config.hyper_parameters.n_fft
        self.hop_length = config.hyper_parameters.hop_size
        self.output_norm = config.data_server.output_norm

        self.target = target
        self.which_set = which_set

        self.n_jobs = config.data_server.n_jobs

        self.config = config

        # load dataset into instance 
        self._load()

    def _load(self):

        if os.path.exists(self.config.paths.path_map):
            self._path_map = pkl.load(open(self.config.paths.path_map))

            if hasattr(self.config.paths.meta_data.splits, self.target):
                split_fn = eval(
                    'self.config.paths.meta_data.splits.{}'.format(self.target)
                )
                split_fn = os.path.join(
                    self.config.paths.meta_data.root, split_fn)

                self.internal_idx = joblib.load(split_fn)[self.which_set]
            else:
                raise IOError(
                    '[ERROR] cannot load split file!')

            if hasattr(self.config.paths.meta_data.targets, self.target):
                target_fn = eval(
                    'self.config.paths.meta_data.targets.{}'.format(self.target)
                )
                target_fn = os.path.join(
                    self.config.paths.meta_data.root, target_fn)

                target = joblib.load(target_fn)

                target_ref = {v:k for k,v in enumerate(target['tids'])}
                self.Y = target['item_factors']

                # output standardization
                if self.output_norm:
                    self.out_sclr = StandardScaler()
                else:
                    self.out_sclr = FunctionTransformer(func=lambda x:x)
                self.Y = self.out_sclr.fit_transform(self.Y)

                # filter out error entries (no data)
                incl = filter(
                    lambda t:t in self._path_map,
                    self.internal_idx
                )

                self.Y = self.Y[map(lambda x:target_ref[x],incl)]
                self.internal_idx = map(lambda x:x,incl)

                if self.Y.shape[0] != len(self.internal_idx):
                    raise ValueError('length bet. index and targets are not\
                                     consistant!')
            else:
                self.Y = None

        else:
            raise IOError("Can't find 'config.paths.path_map'!")


    @property
    def num_examples(self):
        """
        """
        return len(self.internal_idx)

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError

        # (batch,2,sr*length)
        try:
            batch_sz = len(request)
            length_sp = int(self.sr * self.length)

            if self.target != 'self':
                # convert index
                request_fn = map(
                    lambda x:
                    os.path.join(
                        self.config.paths.audio.root,
                        self._path_map[
                            self.internal_idx[x]
                        ]
                    ),
                    request
                )

                signal = pmap(
                    partial(
                        load_audio,
                        sr=self.sr
                    ),
                    request_fn,
                    n_jobs=self.n_jobs
                )
                signal = [s[0] for s in signal]
                signal, mask = zero_pad_signals(signal)

                # fetch target
                target = map(lambda ix:self.Y[ix],request)
                data = filter(
                    lambda y:y[1].sum() > length_sp,
                    zip(signal, mask, target)
                )
                X = map(lambda x:x[0], data)
                M = map(lambda x:x[1], data)
                Y = map(lambda x:x[2], data)

                # prepare sub batch
                X, Y = prepare_sub_batches(
                    self.sub_batch_sz, length_sp,
                    X, M, Y)
            else:
                # get index list
                triplet = sample_matcher_idx(
                    request, self.internal_idx)

                # make hash for batch elements
                uniq_idx = list(set(list(
                    chain.from_iterable(
                        map(lambda x:(x[0], x[1]), triplet)
                    )
                )))
                uniq_hash = {v:k for k, v in enumerate(uniq_idx)}

                # convert index into path
                uniq_paths = map(
                    lambda x:
                    os.path.join(
                        self.config.paths.audio.root,
                        self._path_map[
                            self.internal_idx[x]]
                    ),
                    uniq_idx
                )

                signal = pmap(
                    partial(
                        load_audio,
                        sr=self.sr
                    ),
                    uniq_paths,
                    n_jobs=self.n_jobs
                )
                signal = [s[0] for s in signal]
                signal, mask = zero_pad_signals(signal)

                # list of (2,sr*length)
                data = filter(
                    lambda x:x[1].sum() > length_sp,
                    zip(signal, mask, uniq_idx)
                )
                survivors = set(map(lambda x:x[2], data))
                data = {d[2]:(d[0], d[1]) for d in data}

                # assign databatch into original order
                Xl, Xr, Ml, Mr, Y = [], [], [], [], []
                for d in triplet:
                    if (d[0] not in survivors) or (d[1] not in survivors):
                        continue
                    else:
                        Xl.append(data[d[0]][0])
                        Xr.append(data[d[1]][0])
                        Ml.append(data[d[0]][1])
                        Mr.append(data[d[1]][1])
                        Y.append(d[2])

                # prepare sub batch
                Xl, Y = prepare_sub_batches(
                    self.sub_batch_sz, length_sp,
                    Xl, Ml, Y)
                Xr, _ = prepare_sub_batches(
                    self.sub_batch_sz, length_sp,
                    Xr, Mr)
                X = np.array([Xl, Xr]).transpose(1,0,2,3)
                y = np.eye(2)
                Y = y[Y.ravel().astype(int).tolist()]

            print(X.shape,Y.shape)

        except Exception as e:
            traceback.print_exc()
            # raise Exception
            return -1, -1, request
        else:
            return X, Y, request

    def _get_feature(self,X):
        """"""
        if self.source == 'raw':
            return X
        else:
            raise ValueError(
                '{} is not supported feature type!'.format(
                    self.source)
            )


def launch_data_server(dataset, port, config):
    """
    """
    n_items = dataset.num_examples
    batch_sz = config.hyper_parameters.batch_size
    it_schm = ShuffledScheme(n_items, batch_sz)
    data_stream = DataStream(
        dataset=dataset,
        iteration_scheme=it_schm
    )

    try:
        start_server(
            data_stream,
            port=port,
            hwm=config.data_server.hwm
        )
    except KeyboardInterrupt as ke:
        print(ke)
    finally:
        data_stream.close()

def initiate_data_server(target, which_set, port, config_fn):
    """
    """
    config = load_config(config_fn)

    print('Initialize Data Server...')
    dataset = MSD(
        target=target,
        which_set=which_set,
        config=config,
    )
    launch_data_server(dataset, port, config)

if __name__ == "__main__":
    fire.Fire(initiate_data_server)
