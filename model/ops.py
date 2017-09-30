import numpy as np
import theano
from theano import tensor as T

import lasagne
from lasagne import layers as L

import librosa

import scipy

# function is from Kapre
def _get_stft_kernels(n_dft, window=scipy.signal.hann, keras_ver='new'):
    '''Return dft kernels for real/imagnary parts assuming
        the input signal is real.
    An asymmetric hann window is used (scipy.signal.hann).
    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        Number of dft components.
    keras_ver : string, 'new' or 'old'
        It determines the reshaping strategy.
    Returns
    -------
    dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    * nb_filter = n_dft/2 + 1
    * n_win = n_dft
    '''
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = n_dft / 2 + 1

    # prepare DFT filters
    timesteps = range(n_dft)
    w_ks = [(2 * np.pi * k) / float(n_dft) for k in xrange(n_dft)]
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps]
                                  for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps]
                                  for w_k in w_ks])

    # windowing DFT filters
    dft_window = window(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    if keras_ver == 'old':  # 1.0.6: reshape filter e.g. (5, 8) -> (5, 1, 8, 1)
        dft_real_kernels = dft_real_kernels[:nb_filter]
        dft_imag_kernels = dft_imag_kernels[:nb_filter]
        dft_real_kernels = dft_real_kernels[:, np.newaxis, :, np.newaxis]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, :, np.newaxis]
    else:
        dft_real_kernels = dft_real_kernels[:nb_filter].transpose()
        dft_imag_kernels = dft_imag_kernels[:nb_filter].transpose()
        dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    return dft_real_kernels.astype(np.float32), dft_imag_kernels.astype(np.float32)

def _log_amp(x):
    """
    """
    log_spec = 10 * T.log(T.maximum(x, 1e-10))/T.log(10)
    log_spec = log_spec - T.max(log_spec)  # [-?, 0]
    log_spec = T.maximum(log_spec, -80.0)  # [-80, 0]

    return log_spec

def fft_op(input,n_fft,stride,win=scipy.signal.hann,log_amp=True):
    """
    """
    dft_kernel_r, dft_kernel_i = _get_stft_kernels(n_fft,win,'old')

    X_real = T.nnet.conv2d(input,dft_kernel_r,subsample=stride)
    X_imag = T.nnet.conv2d(input,dft_kernel_i,subsample=stride)

    Xm = (X_real**2 + X_imag**2)**0.5

    if log_amp:
        Xm = _log_amp(Xm)

    return Xm

def test_fft_op(y,n_fft=2048,hop_length=512):
    """
    """
    x = T.tensor4('signal')
    Xm = fft_op(x,n_fft,stride=(hop_length,1),log_amp=True)
    f_mag = theano.function([x],Xm)
    return f_mag(y.astype(np.float32))

def mel_op(input,sr,n_fft,n_mel,stride,log_amp=True):
    """
    """
    mel_kernel = librosa.filters.mel(sr,n_fft,n_mels=n_mel)[:,None,:,None]
    mel_kernel = mel_kernel.astype(np.float32)
    dft = fft_op(input,n_fft,stride,log_amp=False).dimshuffle((0,3,1,2))
    mel = T.nnet.conv2d(dft,mel_kernel)

    if log_amp:
        mel = _log_amp(mel)

    return mel

def test_mel_op(y,sr,n_fft=2048,n_mel=128,hop_length=512):
    """
    """
    x = T.tensor4('signal')
    Xm = mel_op(x,sr,n_fft,n_mel,stride=(hop_length,1),log_amp=True)
    f_mag = theano.function([x],Xm)
    return f_mag(y.astype(np.float32))

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitude
    code is derived from ibab's wavenet github
    '''
    mu = float(quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988)
    # Minimum operation is here to deal with rare large amplitudes caused
    # my resampling
    safe_audio_abs = T.minimum(abs(audio), 1.0)
    magnitude = T.log1p(mu * safe_audio_abs) / T.log1p(mu)
    signal = T.sgn(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return T.cast((signal + 1) / 2 * mu + 0.5, 'int32')

def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1]
    signal = 2 * (T.cast(output, 'float32') / mu) - 1
    # Perform inverse of mu-law transform.
    manitude = (1. / mu) * ((1. + mu)**abs(signal) - 1)
    return T.sgn(signal) * magnitude
