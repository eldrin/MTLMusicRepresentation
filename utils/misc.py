import os
import sys
import time

import subprocess
import tempfile
import logging
import tensorboard_logger as tblog

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from scipy import sparse as sp
# from scipy.ndimage import zoom
from scipy.signal import lfilter

from sklearn.externals import joblib

import theano

import librosa
import soundfile as sf
import sox

from tqdm import tqdm

import namedtupled
import json

sys.setrecursionlimit(40000)


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
    if any(
        [(len(config['target']) != len(config['hyper_parameters']['n_out'])),
         (len(config['target']) != len(config['hyper_parameters']['out_act'])),
            (len(config['hyper_parameters']['n_out']) !=
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
            config['paths']['meta_data']['size'][target]['train'] = \
                len(split['train'])
            config['paths']['meta_data']['size'][target]['valid'] = \
                len(split['valid'])

    return namedtupled.map(config)


def get_layer(net, layer_name):
    """"""
    # layers = L.get_all_layers(net['IO'])
    # return filter(lambda x:x.name == layer_name, layers)[0]
    return net[layer_name]


def get_in_shape(config):
    """"""
    if hasattr(config.hyper_parameters, 'n_ch'):
        n_ch = config.hyper_parameters.n_ch
    else:
        n_ch = 2  # backward compatibility (will be removed)

    # win_sz = config.hyper_parameters.n_fft
    hop_sz = config.hyper_parameters.hop_size
    sr = config.hyper_parameters.sample_rate
    length = config.hyper_parameters.patch_length
    remaining = int(sr*length) % hop_sz
    sig_len = int(sr*length) - remaining

    if config.hyper_parameters.input == 'signal':
        return (None, n_ch, sig_len)

    elif config.hyper_parameters.input == 'melspec':
        dur = int(sig_len / hop_sz) + 1
        return (None, n_ch, dur, 128)


def get_loggers(config):
    """"""
    # make name string here
    # make target string
    target_string = '_'.join(config.target)
    fn = config.paths.file_name.format(target_string)

    log_fn = os.path.join(config.paths.log, fn + '.log')

    # init standard logger
    pylog = get_py_logger(log_fn)

    # init tensorboard logger
    tblog.configure(os.path.join(config.paths.tblog, fn))

    return pylog, tblog


def get_py_logger(fn):
    """"""
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s][%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(fn)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger


def preemphasis(signal):
    return lfilter([1, -0.70], 1, signal)


def deemphasis(signal):
    return lfilter([1, 0.70], 1, signal)


def load_mel(fn):
    """
    """
    # a = np.load(fn, mmap_mode='r')
    a = np.load(fn)
    return a[0]


def load_audio(fn, sr=None):
    """
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpf:
            subprocess.call(
                ['mpg123', '-w', tmpf.name, '-q', fn]
            )

            y, sr_file = sf.read(
                tmpf.name, always_2d=True, dtype='float32')

            if y.size < 100:
                raise ValueError('File is too small')

            # transpose & crop
            y = y.T

            # resampling for outliers
            if (sr is not None) and (int(sr_file) != int(sr)):
                # print('resample!')
                y = librosa.resample(
                    y, sr_file, sr,
                    res_type='kaiser_fast'
                )

            # process mono
            if y.shape[0] < 2:
                y = np.repeat(y, 2, axis=0)

    except Exception as e:
        print(e)
        return None, None
    else:
        return y, sr


def load_audio_batch(fn, sr, mono=False, dur=5.):
    """
    """
    if not os.path.exists(fn):
        print('file not exists!')
        return None

    try:
        length = sox.file_info.duration(fn)
        sr_org = sox.file_info.sample_rate(fn)
        # n_samples = sox.file_info.num_samples(fn)
        n_ch = sox.file_info.channels(fn)

        if length < dur:
            # print('{} too short!(1) : {} / {}'.format(fn, length, dur))
            return None

        if n_ch < 2:
            mono = True
        else:
            mono = False

        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpf:
            subprocess.call(
                ['mpg123', '-w', tmpf.name, '-q', fn]
            )

            # st_sec = np.random.choice(int((length-dur)*10))/10.
            # st = int(st_sec * sr_org)
            st = np.random.choice(int((length - dur) * sr_org))
            n_frames = int(dur * sr_org)

            y, sr_file = sf.read(
                tmpf.name, always_2d=True, dtype='float32')
            y = y[st:st+n_frames]

            # transpose & crop
            y = y.T

            # resampling for outliers
            if int(sr_file) != int(sr):
                y = librosa.resample(
                    y, sr_file, sr,
                    res_type='kaiser_fast'
                )

            # process mono
            if mono:
                y = np.repeat(y[None, :], 2, axis=0)

        # if length is reasonably shorter than delivery length
        # pad it with zeros
        trg_len = sr * dur
        src_len = y.shape[1]
        thresh = 0.95 * trg_len
        if src_len < thresh:
            # print('too short! (2)')
            return None

        elif src_len < trg_len and src_len >= thresh:
            npad = int(trg_len - src_len)
            y = np.concatenate(
                [y, np.zeros((y.shape[0], npad))],
                axis=-1
            )

    except sox.SoxiError:
        # print(ee)
        return None
    except sox.SoxError:
        return None
    except Exception as e:
        print(e)
        return None
    else:
        return y


def zero_pad_signals(signal):
    """"""
    n_ch = signal[0].shape[0]
    time_dim = 1
    longest_len = np.max(
        [s.shape[time_dim] if s is not None else 0 for s in signal])

    if signal[0].ndim == 3:
        feat_dim = signal[0].shape[-1]
        dim = (len(signal), n_ch, longest_len, feat_dim)
    elif signal[0].ndim == 2:
        dim = (len(signal), n_ch, longest_len)

    S = np.zeros(dim, dtype=np.float32)
    M = np.zeros(dim[:3], dtype=np.int8)  # mask
    for i, s in enumerate(signal):
        if s is None:
            continue
        S[i, :, :s.shape[time_dim]] = s
        M[i, :, :s.shape[time_dim]] = 1

    return S, M


def prepare_sub_batches(n, dur, signal, mask, target=None):
    """"""
    # prepare n subbatch from batch
    batch_sz = len(signal)
    n_ch = signal[0].shape[0]
    sig_len = signal[0].shape[1]

    if target is None:
        target_dim = 1
        target = -np.ones((batch_sz, target_dim))  # dummy values
    else:
        if isinstance(target[0], int):
            target_dim = 1
        else:
            target_dim = target[0].shape[-1]

    if signal[0].ndim == 3:
        X = np.zeros(((n * batch_sz), n_ch, dur, signal[0].shape[-1]))
    elif signal[0].ndim == 2:
        X = np.zeros(((n * batch_sz), n_ch, dur))
    Y = np.zeros(((n * batch_sz), target_dim))
    for i, x, m, y in zip(range(batch_sz), signal, mask, target):
        for j in xrange(n):
            st = np.random.choice(sig_len - dur)
            X[j * batch_sz + i] = x[:, st:st+dur]
            Y[j * batch_sz + i] = y
    return X, Y


def pmap(function, array, n_jobs=16, use_kwargs=False, front_num=3,
         verbose=False):
    """
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a)
                 for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension.
    # This is useful for benchmarking and debugging.
    if n_jobs == 1:
        if verbose:
            array_remain = tqdm(array[front_num:])
        else:
            array_remain = array[front_num:]
        return front + [function(**a) if use_kwargs else function(a) for a in
                        array_remain]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        # kwargs = {
        #     'total': len(futures),
        #     'unit': 'it',
        #     'unit_scale': True,
        #     'leave': True
        # }
        # Print out the progress as tasks complete
        if verbose:
            compl_futures = tqdm(as_completed(futures))
        else:
            compl_futures = as_completed(futures)
        for f in compl_futures:
            pass
    out = []
    # Get the results from the futures.
    if verbose:
        enum_futures = tqdm(enumerate(futures))
    else:
        enum_futures = enumerate(futures)
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
            def maybe_to_gpu(x): return x
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


def test_signal_batching():
    sr = 22050
    min_audio_len = 30 * sr
    max_audio_len = 60 * sr
    dur = int(2.5 * sr)
    n = 64  # num sample in batch
    m = 20  # num sub batch in batch

    signal = [np.random.rand(n_) for n_
              in np.random.randint(
                  min_audio_len, max_audio_len, size=n)]
    signal, mask = zero_pad_signals(signal)

    target = [np.random.rand(20) for _ in xrange(n)]
    target = [t / t.sum() for t in target]  # normalize

    t = time.time()
    X, Y = prepare_sub_batches(m, dur, signal, mask, target)
    print('took {:.2f} second'.format(time.time() - t))
    print(X.shape, Y.shape)


def triplet2sparse(triplet, doc_hash=None, term_hash=None):
    """"""
    val = map(lambda x: x[2], triplet)
    if doc_hash is None:
        row = map(lambda x: x[0], triplet)
        n_row = len(set(row))
    else:
        row = map(lambda x: doc_hash[x[0]], triplet)
        n_row = len(doc_hash)

    if term_hash is None:
        col = map(lambda x: x[1], triplet)
        n_col = len(set(col))
    else:
        col = map(lambda x: term_hash[x[1]], triplet)
        n_col = len(term_hash)

    A = sp.coo_matrix(
        (val, (row, col)), shape=(n_row, n_col), dtype=int
    )  # (n_items, n_words)

    return A


def load_test_audio(config):
    """"""
    fn = config.paths.test_audio
    sr = config.hyper_parameters.sample_rate
    return librosa.load(fn, sr=sr)


if __name__ == "__main__":
    test_signal_batching()
