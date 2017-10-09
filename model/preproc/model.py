import lasagne
from lasagne import layers as L

import theano
from theano import tensor as T

from ..custom_layer import STFTLayer, MelSpecLayer

class MelSpectrogramGPU:
    """"""
    def __init__(self, n_ch, sr, n_fft, hop_size):
        """"""
        self.n_ch = n_ch
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size

        # init & compile model
        l_in = L.InputLayer((None, n_ch, None))
        reshape = L.ReshapeLayer(l_in, ([0], [1], [2], 1))
        stft = STFTLayer(reshape, n_ch=n_ch, n_fft=n_fft, hop_size=hop_size,
                         log_amplitude=False)
        mel = MelSpecLayer(stft, sr, n_fft, log_amplitude=True)
        out = L.get_output(mel, deterministic=True)

        self._process = theano.function(
            [l_in.input_var], out, allow_input_downcast=True)

    def process(self, x):
        """"""
        return self._process(x)
