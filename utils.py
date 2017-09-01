import os
import sys
sys.setrecursionlimit(40000)

import copy
import subprocess
import tempfile
import logging
import tensorboard_logger as tblog

from concurrent.futures import ProcessPoolExecutor, as_completed, wait

import numpy as np
import pandas as pd

from scipy import sparse as sp
from scipy.ndimage import zoom
from scipy.signal import lfilter

from sklearn.externals import joblib

import theano
from theano import tensor as T

import lasagne
from lasagne import layers as L

from fuel.streams import ServerDataStream

import librosa
import soundfile as sf
import sox

from tqdm import tqdm

import namedtupled
config = namedtupled.json(path=open('config.json','r'), name='config')


def get_in_shape(config):
    """"""
    win_sz = config.hyper_parameters.n_fft
    hop_sz = config.hyper_parameters.hop_size
    sr = config.hyper_parameters.sample_rate
    length = config.hyper_parameters.patch_length

    remaining = int(sr*length) % hop_sz
    sig_len = int(sr*length) - remaining

    return (None, 2, sig_len)


def open_datastream(config, is_train=True):
    """
    """
    if is_train:
        port = config.data_server.train_port
    else:
        port = config.data_server.valid_port

    host = config.data_server.host
    hwm = config.data_server.hwm

    return ServerDataStream(
        sources=('raw'), produces_examples=True,
        port=port, host=host, hwm=hwm
    )


def get_loggers(config):
    """"""
    fn = config.paths.file_name.format(config.target)
    fn += '.log'
    log_fn = os.path.join(config.paths.log, fn)

    # init standard logger
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s][%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(log_fn)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # init tensorboard logger
    tblog.configure(os.path.join(config.paths.tblog,fn))

    return rootLogger, tblog


def load_check_point(network, config):
    """
    """
    fns = get_check_point_fns(config)
    it = 0

    if fns['param'] is not None and os.path.exists(fns['param']):
        try:
            print('Loadong pre-trained weight...')

            with np.load(fns['param']) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]

            lasagne.layers.set_all_param_values(network, param_values)
            it = joblib.load(fns['state'])['iter']
        except Exception as e:
            print(e)
            print('Cannot load parameters!')
    else:
        print('Cannot find parameters!')

    return it, network


def save_check_point(it, network, config):
    """
    """
    fns = get_check_point_fns(config)
    config_dict = namedtupled.reduce(config)

    np.savez(fns['param'], *lasagne.layers.get_all_param_values(network))
    joblib.dump({'iter':it, 'config':config_dict}, fns['state'])


def get_check_point_fns(config):
    """"""
    fns = {}
    fns['param'] = None
    fns['state'] = None

    dump_root = config.paths.model
    fname = config.paths.file_name.format(config.target)
    suffix_param = '_param.npz'
    suffix_state = '_state.dat.gz'

    try:
        fns['param'] = os.path.join(dump_root, fname + suffix_param)
        fns['state'] = os.path.join(dump_root, fname + suffix_state)

    except Exception as e:
        raise e # TODO: elaborate this

    return fns


def preemphasis(signal):
    return lfilter([1, -0.70], 1, signal)


def deemphasis(signal):
    return lfilter([1, 0.70], 1, signal)


def prepare_X(x,preproc,n_fft=1024,hop_sz=256):
    """ Prepare input for 2D CNN """

    # shape of x should be
    # (n_ch, n_sample-(5s))
    X = map(
        lambda x:preproc.transform(x.T).T,
        map(
            lambda ch:
            np.log10(
                np.abs(
                    librosa.stft(
                        ch,n_fft=n_fft,
                        hop_length=hop_sz
                    )
                ) + EPS
            ),
            x
        )
    )
    X = np.array(X)[None,:,:]

    return X


def load_audio(fn,sr,mono=False,dur=5.):
    """
    """
    if not os.path.exists(fn):
        return None

    try:
        length = sox.file_info.duration(fn)
        sr_org = sox.file_info.sample_rate(fn)
        n_samples = sox.file_info.num_samples(fn)
        n_ch = sox.file_info.channels(fn)

        if length < dur:
            return None

        if n_ch < 2:
            mono = True
        else:
            mono = False

        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpf:
            subprocess.call(
                ['mpg123','-w',tmpf.name,'-q',fn]
            )

            st_sec = np.random.choice(int((length-dur)*10))/10.
            st = int(st_sec * sr_org)
            n_frames = int(dur * sr_org)

            y,sr_file = sf.read(
                tmpf.name, frames=n_frames, start=st,
                always_2d=True, dtype='float32')

            # transpose & crop
            y = y.T

            # resampling for outliers
            if sr_file != sr:
                y = librosa.resample(
                    y,sr_file,sr,
                    res_type='kaiser_fast'
                )

            # process mono
            if mono:
                y = np.repeat(y[None,:],2,axis=0)

        # if length is reasonably shorter than delivery length
        # pad it with zeros
        trg_len = sr * dur
        src_len = y.shape[1]
        thresh = 0.75 * trg_len
        if src_len < thresh:
            return None

        elif src_len < trg_len and src_len >= thresh:
            npad = trg_len - src_len
            y = np.concatenate(
                [y, np.zeros((y.shape[0],npad))],
                axis=-1
            )

    except sox.SoxiError as ee:
        # print(ee)
        return None
    except sox.SoxError:
        return None
    except Exception as e:
        print(e)
        return None
    else:
        return y


def pmap(function, array, n_jobs=16, use_kwargs=False, front_num=3):
    """
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in as_completed(futures):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)


class GuidedBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)

