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

import librosa
import soundfile as sf
import sox

from tqdm import tqdm

import namedtupled

import json

def load_config(fn):
    """"""
    config = json.load(open(fn, 'r'))

    if not isinstance(config['target'], list):
        config['target'] = [config['target']]

    if not isinstance(config['hyper_parameters']['n_out'], list):
        config['hyper_parameters']['n_out'] = \
                [config['hyper_parameters']['n_out']]

    if not isinstance(config['hyper_parameters']['out_act'], list):
        config['hyper_parameters']['out_act'] = \
                [config['hyper_parameters']['out_act']]

    # check if targets, outs, acts have same length
    if any([(len(config['target']) != len(config['hyper_parameters']['n_out'])),
            (len(config['target']) != len(config['hyper_parameters']['out_act'])),
            (len(config['hyper_parameters']['n_out']) != \
             len(config['hyper_parameters']['out_act']))]):
        raise ValueError(
            '[ERROR] target, n_out, act_out must have same length!')

    # get dataset sizes
    meta_data_root = config['paths']['meta_data']['root']
    split_fns = config['paths']['meta_data']['splits']
    config['paths']['meta_data']['size'] = {}
    for target in config['target']:
        split_fn = os.path.join(meta_data_root, split_fns[target])
        config['paths']['meta_data']['size'][target] = {}
        if not os.path.exists(split_fn):
            raise ValueError('[ERROR] target is not existing')
        else:
            split = joblib.load(split_fn)
            config['paths']['meta_data']['size'][target]['train'] = len(split['train'])
            config['paths']['meta_data']['size'][target]['valid'] = len(split['valid'])

    return namedtupled.map(config)


def get_layer(net, layer_name):
    """"""
    layers = L.get_all_layers(net['IO'])
    return filter(lambda x:x.name == layer_name, layers)[0]


def get_in_shape(config):
    """"""
    win_sz = config.hyper_parameters.n_fft
    hop_sz = config.hyper_parameters.hop_size
    sr = config.hyper_parameters.sample_rate
    length = config.hyper_parameters.patch_length

    remaining = int(sr*length) % hop_sz
    sig_len = int(sr*length) - remaining

    return (None, 2, sig_len)


def get_loggers(config):
    """"""
    # make name string here
    # make target string
    target_string = '_'.join(config.target)
    fn = config.paths.file_name.format(target_string)

    log_fn = os.path.join(config.paths.log, fn + '.log')

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

def preemphasis(signal):
    return lfilter([1, -0.70], 1, signal)


def deemphasis(signal):
    return lfilter([1, 0.70], 1, signal)

def load_audio(fn, sr=None):
    """
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpf:
            subprocess.call(
                ['mpg123','-w',tmpf.name,'-q',fn]
            )

            y, sr_file = sf.read(
                tmpf.name, always_2d=True, dtype='float32')

            # transpose & crop
            y = y.T

            # resampling for outliers
            if (sr is not None) and (sr_file != sr):
                y = librosa.resample(
                    y, sr_file, sr,
                    res_type='kaiser_fast'
                )

            # process mono
            if mono:
                y = np.repeat(y[None,:],2,axis=0)

    except Exception as e:
        print(e)
        return None, None
    else:
        return y, sr

def load_audio_batch(fn,sr,mono=False,dur=5.):
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


def pmap(function, array, n_jobs=16, use_kwargs=False, front_num=3,
         verbose=False):
    """
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        if verbose: array_remain = tqdm(array[front_num:])
        else: array_remain = array[front_num:]
        return front + [function(**a) if use_kwargs else function(a) for a in
                        array_remain]
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
        if verbose: compl_futures = tqdm(as_completed(futures))
        else: compl_futures = as_completed(futures)
        for f in compl_futures:
            pass
    out = []
    #Get the results from the futures. 
    if verbose: enum_futures = tqdm(enumerate(futures))
    else: enum_futures = enumerate(futures)
    for i, future in enum_futures:
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

