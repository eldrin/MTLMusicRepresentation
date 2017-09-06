import os
import argparse
from collections import OrderedDict
import cPickle as pkl
import sqlite3

import numpy as np
import pandas as pd

from fuel.schemes import ShuffledScheme,SequentialScheme
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.server import start_server
from fuel.transformers import Transformer
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, FunctionTransformer

import h5py
from functools import partial
import tempfile

import librosa
import fire

from utils.misc import pmap, load_audio, load_config

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

            if self.target != 'self':

                split_fn = eval(
                    'self.config.paths.meta_data.splits.{}'.format(self.target)
                )
                split_fn = os.path.join(
                    self.config.paths.meta_data.root, split_fn)

                target_fn = eval(
                    'self.config.paths.meta_data.targets.{}'.format(self.target)
                )
                target_fn = os.path.join(
                    self.config.paths.meta_data.root, target_fn)

                self.internal_idx = joblib.load(split_fn)[self.which_set]
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
                    raise ValueError('length bet. index and targets are not consistant!')

            elif self.target=='self':
                self.internal_idx = self._path_map.keys()

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

            # fetch signal
            signal = pmap(
                partial(
                    load_audio,
                    sr=self.sr,
                    dur=self.length
                ),
                request_fn,
                n_jobs=self.n_jobs
            )

            if self.target!='self':
                # fetch target
                target = map(lambda ix:self.Y[ix],request)
                data = filter(
                    lambda y:y[0] is not None,
                    zip(signal,target)
                )
                X = np.array(map(lambda x:x[0],data)).astype(np.float32)
                X = self._get_feature(X)
                Y = np.array(map(lambda x:x[1],data)).astype(np.float32)

            else:
                # list of (2,sr*length)
                X = filter(lambda y:y[0] is not None,signal)
                X = np.array(X).astype(np.float32)
                X = self._get_feature(X)
                Y = -1. # null data

            print(X.shape,Y.shape)

        except Exception as e:
            print(e)
            print([x.shape for x in X])
            # raise Exception
            return -1,-1,request
        else:
            return X,Y,request

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