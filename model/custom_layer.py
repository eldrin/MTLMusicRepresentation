import theano
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import (AbstractConv3d,
                                              AbstractConv3d_gradWeights,
                                              AbstractConv3d_gradInputs)

from theano import config

import lasagne
from lasagne.layers import get_all_layers
from lasagne.layers import MergeLayer, Layer, InputLayer, InverseLayer
from lasagne.layers import NonlinearityLayer, BiasLayer
from lasagne.layers import DropoutLayer, GaussianNoiseLayer

from lasagne import init
from lasagne.layers.recurrent import Gate
from lasagne.layers.conv import BaseConvLayer
from lasagne import nonlinearities
import numpy as np
import scipy

from ops import _get_stft_kernels, _log_amp

from lasagne.layers.conv import conv_output_length
from lasagne.utils import as_tuple

class STFTLayer(Layer):
    """Custom Layer for FFT
        outputs magnitude STFT
        input shape is same as Conv2DLayer
    """
    def __init__(
        self, incoming, window=scipy.signal.hann, n_ch=2,
        n_fft=2048, hop_size=512, log_amplitude=True, **kwargs):
        """
        """
        super(STFTLayer, self).__init__(incoming, **kwargs)

        n = 2 # 2D convolution

        if n_ch > 2 or n_ch < 1:
            raise ValueError(
                "n_ch should be either 1 (mono) or 2 (stereo)"
            )

        self.n_ch = n_ch
        self.window = window
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.log_amp = log_amplitude

        self.filter_size = as_tuple((n_fft,1), n, int)
        self.stride = as_tuple((hop_size,1), n, int)

        # dft kernels for real and imaginary domain
        W_r, W_i = _get_stft_kernels(
            n_fft, window=window, keras_ver='old')

        self.W_r = self.add_param(
            W_r, shape=W_r.shape, name='DFT_real_kernel',
            trainable=False, regularizable=False)
        self.W_i = self.add_param(
            W_i, shape=W_i.shape, name='DFT_image_kernel',
            trainable=False, regularizable=False)

    def get_output_shape_for(self, input_shape):
        pad = (0,0)
        batchsize = input_shape[0]
        shape_raw = ((batchsize, self.n_fft/2+1) +
            tuple(conv_output_length(input, filter, stride, p)
                for input, filter, stride, p
                in zip(input_shape[2:], self.filter_size,
                    self.stride, pad)))

        if self.n_ch == 2:
            return (shape_raw[0], 2, shape_raw[1], shape_raw[2])
        elif self.n_ch == 1:
            return (shape_raw[0], 1, shape_raw[1], shape_raw[2])


    def get_output_for(self, input, **kwargs):

        if self.n_ch == 2:
            left_ch = input[:,0,:,:][:,None,:,:]
            right_ch = input[:,1,:,:][:,None,:,:]

            X_real_l = T.nnet.conv2d(left_ch,self.W_r,subsample=self.stride)
            X_real_r = T.nnet.conv2d(right_ch,self.W_r,subsample=self.stride)
            X_real = T.concatenate([X_real_l, X_real_r], axis=-1)

            X_imag_l = T.nnet.conv2d(left_ch,self.W_i,subsample=self.stride)
            X_imag_r = T.nnet.conv2d(right_ch,self.W_i,subsample=self.stride)
            X_imag = T.concatenate([X_imag_l, X_imag_r], axis=-1)

        elif self.n_ch == 1:
            X_real = T.nnet.conv2d(input,self.W_r,subsample=self.stride)
            X_imag = T.nnet.conv2d(input,self.W_i,subsample=self.stride)

        # magnitude STFT
        Xm = (X_real**2 + X_imag**2)**0.5

        # log amplitude
        if self.log_amp:
            Xm = _log_amp(Xm)

        # return T.cast(Xm.dimshuffle(0,3,1,2),config.floatX)
        return Xm.dimshuffle(0,3,1,2)


class LayerNormLayer(Layer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(LayerNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all
            axes = tuple(range(0, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        mean = input_mean
        inv_std = input_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized


class LNGRULayer(MergeLayer):
    r"""
    .. math ::
        r_t = \sigma_r(LN(x_t, W_{xr}; \gamma_{xr}, \beta_{xr}) + LN(h_{t - 1}, W_{hr}; \gamma_{xr}, \beta_{xr}) + b_r)\\ \
        u_t = \sigma_u(LN(x_t, W_{xu}; \gamma_{xu}, \beta_{xu}) + LN(h_{t - 1}, W_{hu}; \gamma_{xu}, \beta_{xu})+ b_u)\\ \
        c_t = \sigma_c(LN(x_t, W_{xc}; \gamma_{xc}, \beta_{xc}) + r_t \odot (LN(h_{t - 1}, W_{hc}); \gamma_{xc}, \beta_{xc}) + b_c)\\ \
        h_t = (1 - u_t) \odot h_{t - 1} + u_t \odot c_t \

    Notes
    -----

    .. math::
        LN(z;\alpha, \beta) = \frac{(z-\mu)}{\sigma} \odot \alpha + \beta

    """
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 alpha_init=init.Constant(1.0),
                 beta_init=init.Constant(0.0),
                 normalize_hidden_update=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LNGRULayer, self).__init__(incomings, **kwargs)

        # # If the provided nonlinearity is None, make it linear
        # if nonlinearity is None:
        #     self.nonlinearity = nonlinearities.identity
        # else:
        #     self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.normalize_hidden_update = normalize_hidden_update
        self._eps = 1e-5

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(self.alpha_init, (num_units,),
                                   name="alpha_in_to_{}".format(gate_name)),
                    self.add_param(self.beta_init, (num_units,),
                                   name="beta_in_to_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(self.alpha_init, (num_units,),
                                   name="alpha_hid_to_{}".format(gate_name)),
                    self.add_param(self.beta_init, (num_units,),
                                   name="beta_hid_to_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.alpha_in_to_updategate, self.beta_in_to_updategate,
         self.alpha_hid_to_updategate, self.beta_hid_to_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate, 'updategate')

        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.alpha_in_to_resetgate, self.beta_in_to_resetgate,
         self.alpha_hid_to_resetgate, self.beta_hid_to_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.alpha_in_to_hidden_update, self.beta_in_to_hidden_update,
         self.alpha_hid_to_hidden_update, self.beta_hid_to_hidden_update,
         self.nonlinearity_hidden_update) = add_gate_params(hidden_update, 'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # parameters for Layer Normalization of the cell gate
        if self.normalize_hidden_update:
            self.alpha_hidden_update = self.add_param(
                self.alpha_init, (num_units, ),
                name="alpha_hidden_update")
            self.beta_hidden_update = self.add_param(
                self.beta_init, (num_units, ),
                name="beta_hidden_update", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    # Layer Normalization
    def __ln__(self, z, alpha, beta):
        output = (z - z.mean(-1, keepdims=True)) / T.sqrt(z.var(-1, keepdims=True) + self._eps)
        output = alpha * output + beta
        return output


    def __gru_fun__(self, inputs, **kwargs):
        """
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Stack alphas for input into a (3*num_units) vector
        alpha_in_stacked = T.concatenate(
            [self.alpha_in_to_resetgate, self.alpha_in_to_updategate,
             self.alpha_in_to_hidden_update], axis=0)

        # Stack betas for input into a (3*num_units) vector
        beta_in_stacked = T.concatenate(
            [self.beta_in_to_resetgate, self.beta_in_to_updategate,
             self.beta_in_to_hidden_update], axis=0)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack alphas for hidden into a (3*num_units) vector
        alpha_hid_stacked = T.concatenate(
            [self.alpha_hid_to_resetgate, self.alpha_hid_to_updategate,
             self.alpha_hid_to_hidden_update], axis=0)

        # Stack betas for hidden into a (3*num_units) vector
        beta_hid_stacked = T.concatenate(
            [self.beta_hid_to_resetgate, self.beta_hid_to_updategate,
             self.beta_hid_to_hidden_update], axis=0)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            big_ones = T.ones((seq_len, num_batch, 1))
            input = T.dot(input, W_in_stacked)
            input = self.__ln__(input,
                                T.dot(big_ones, alpha_in_stacked.dimshuffle('x', 0)),
                                beta_in_stacked) + b_stacked

        ones = T.ones((num_batch, 1))
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):
            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked)
                input_n = self.__ln__(input_n,
                                      T.dot(ones, alpha_in_stacked.dimshuffle('x', 0)),
                                      beta_in_stacked) + b_stacked

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)
            hid_input = self.__ln__(hid_input,
                                    T.dot(ones, alpha_hid_stacked.dimshuffle('x', 0)),
                                    beta_hid_stacked)

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            if self.grad_clipping:
                resetgate = theano.gradient.grad_clip(
                    resetgate, -self.grad_clipping, self.grad_clipping)
                updategate = theano.gradient.grad_clip(
                    updategate, -self.grad_clipping, self.grad_clipping)

            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hidden_update(hidden_update)

            if self.normalize_hidden_update:
                hidden_update = self.__ln__(hidden_update,
                                   T.dot(ones, self.alpha_hidden_update.dimshuffle('x', 0)),
                                   self.beta_hidden_update)
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, alpha_hid_stacked, beta_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked, alpha_in_stacked, beta_in_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=False)[0]

        return hid_out


    def get_output_for(self, inputs, **kwargs):
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        hid_out = self.__gru_fun__(inputs, **kwargs)
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class Conv3DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv3d, **kwargs):
        BaseConvLayer.__init__(self, incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=3,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved


class TransposedConv3DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 crop=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 output_size=None, **kwargs):
        # output_size must be set before calling the super constructor
        if (not isinstance(output_size, T.Variable) and
                output_size is not None):
            output_size = as_tuple(output_size, 3, int)
        self.output_size = output_size
        BaseConvLayer.__init__(self, incoming, num_filters, filter_size, stride, crop, untie_biases,
                W, b, nonlinearity, flip_filters, n=3, **kwargs)
        # rename self.pad to self.crop:
        self.crop = self.pad
        del self.pad

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        if self.output_size is not None:
            size = self.output_size
            if isinstance(self.output_size, T.Variable):
                size = (None, None)
            return input_shape[0], self.num_filters, size[0], size[1], size[2]

        # If self.output_size is not specified, return the smallest shape
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_input_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = AbstractConv3d_gradInputs(
            imshp=self.output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if isinstance(self.output_size, T.Variable):
            output_size = self.output_size
        elif any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved


# THERE CODES FROM neosatrapahereje's GITHUB
def build_autoencoder(layer, nonlinearity='same', b=init.Constant(0.)):
    """
    Unfolds a stack of layers into a symmetric autoencoder with tied weights.
    Given a :class:`Layer` instance, this function builds a
    symmetric autoencoder with tied weights.
    Parameters
    ----------
    layer : a :class:`Layer` instance or a tuple
        The :class:`Layer` instance with respect to which a symmetric
        autoencoder is built.
    nonlinearity : 'same', list, callable, or None
        The nonlinearities that are applied to the decoding layer.
        If 'same', each decoder layer has the same nonlinearity as its
        corresponding encoder layer. If a list is provided, it must contain
        nonlinearities for each decoding layer. Otherwise, if a single
        nonlinearity is provided, it is applied to all decoder layers.
        If set to ``None``, all nonlinearities for the decoder layers are set
        to lasagne.nonlinearities.identity.
    b : callable, Theano shared variable, numpy array, list or None
        An initializer for the decoder biases. By default, all decoder
        biases are initialized to lasagne.init.Constant(0.). If a shared
        variable or a numpy array is provided, the shape must match the
        incoming shape (only in case all incoming shapes are the same).
        Additianlly, a list containing initializers for the biases of each
        decoder layer can be provided. If set to ``None``, the decoder
        layers will have no biases, and pass through their input instead.
    Returns
    -------
    layer: :class:`Layer` instance
       The output :class:`Layer` of the symmetric autoencoder with
       tied weights.
    encoder: :class:`Layer` instance
       The code :class:`Layer` of the autoencoder (see Notes)
    Notes
    -----
    The encoder (input) :class:`Layer` is changed using
    `unfold_bias_and_nonlinearity_layers`. Therefore, this layer is not the
    code layer anymore, because it has got its bias and nonlinearity stripped
    off.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> from lasagne.layers import build_autoencoder
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> l2 = DenseLayer(l1, num_units=10)
    >>> l_ae, l2 = build_autoencoder(l2, nonlinearity='same', b=None)
    """

    if isinstance(nonlinearity, (tuple, list)):
        n_idx = 0

    if isinstance(b, (tuple, list)):
        b_idx = 0

    encoder = unfold_bias_and_nonlinearity_layers(layer)
    layers = get_all_layers(encoder)
    autoencoder_layers = [encoder]

    kwargs_b = dict(b=None)
    kwargs_n = dict(nonlinearity=nonlinearities.identity)
    for i, layer in enumerate(layers[::-1]):

        incoming = autoencoder_layers[-1]
        if isinstance(layer, InputLayer):
            continue
        elif isinstance(layer, BiasLayer):
            if b is None:
                kwargs_b = dict(b=None)
            elif isinstance(b, (tuple, list)):
                kwargs_b = dict(b=b[b_idx])
                b_idx += 1
            else:
                kwargs_b = dict(b=b)
        elif isinstance(layer, NonlinearityLayer):
            if nonlinearity == 'same':
                kwargs_n = dict(nonlinearity=layer.nonlinearity)
            elif nonlinearity is None:
                kwargs_n = dict(nonlinearity=nonlinearities.identity)
            elif isinstance(nonlinearity, (tuple, list)):
                kwargs_n = dict(nonlinearity=nonlinearity[n_idx])
                n_idx += 1
            else:
                kwargs_n = dict(nonlinearity=nonlinearity)
        elif isinstance(layer, DropoutLayer):
            a_layer = DropoutLayer(
                incoming=incoming,
                p=layer.p,
                rescale=layer.rescale
            )
            autoencoder_layers.append(a_layer)
        elif isinstance(layer, GaussianNoiseLayer):
            a_layer = GaussianNoiseLayer(
                incoming=incoming,
                sigma=layer.sigma
            )
            autoencoder_layers.append(a_layer)
        else:
            a_layer = InverseLayer(
                incoming=incoming,
                layer=layer
            )
            if hasattr(layer, 'b'):
                a_layer = BiasLayer(
                    incoming=a_layer,
                    **kwargs_b
                )
            if hasattr(layer, 'nonlinearity'):
                a_layer = NonlinearityLayer(
                    incoming=a_layer,
                    **kwargs_n
                )
            autoencoder_layers.append(a_layer)

    return autoencoder_layers, encoder


def unfold_bias_and_nonlinearity_layers(layer):
    """
    Unfolds a stack of layers adding :class:`BiasLayer` and
    :class:`NonlinearityLayer` when needed.
    Given a :class:`Layer` instance representing a stacked network,
    this function adds a :class:`BiasLayer` instance and/or a
    :class:`NonlinearityLayer` instance in between each layer with attributes
    b (bias) and/or nonlinearity, with the same bias and nonlinearity,
    while deleting the bias and or setting the nonlinearity
    of the original layer to the identity
    function.
    Parameters
    ----------
    layer : a :class:`Layer` instance or a tuple
        The :class:`Layer` instance with respect to wich the new
        stacked Neural Network with added :class:`BiasLayer`: and
        class:`NonlinearityLayer` are built.
    Returns
    -------
    layer: :class:`Layer` instance
        The output :class:`Layer` of the symmetric autoencoder with
        tied weights.
    Examples
    --------
    >>> import lasagne
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> from lasagne.layers import BiasLayer, NonlinearityLayer
    >>> from lasagne.layers import unfold_bias_and_nonlinearity_layers
    >>> from lasagne.layers import get_all_layers
    >>> from lasagne.nonlinearities import tanh, sigmoid, identity
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50, nonlinearity=tanh)
    >>> l_out = DenseLayer(l1, num_units=10, nonlinearity=sigmoid)
    >>> l_out = unfold_bias_and_nonlinearity_layers(l_out)
    >>> all_layer_names = [l.__class__.__name__ for l in get_all_layers(l_out)]
    >>> all_layer_names[:4]
    ['InputLayer', 'DenseLayer', 'BiasLayer', 'NonlinearityLayer']
    >>> all_layer_names[4:]
    ['DenseLayer', 'BiasLayer', 'NonlinearityLayer']
    """
    layers = get_all_layers(layer)
    incoming = layers[0]

    for ii, layer in enumerate(layers[1:]):
        layer.input_layer = incoming
        # Check if the layer has a bias
        b = getattr(layer, 'b', None)
        add_bias = False
        # Check if the layer has a nonlinearity
        nonlinearity = getattr(layer, 'nonlinearity', None)
        add_nonlinearity = False
        if b is not None and not isinstance(layer, BiasLayer):
            layer.b = None
            del layer.params[b]
            add_bias = True
        if (nonlinearity is not None and
                not isinstance(layer, NonlinearityLayer) and
                nonlinearity != nonlinearities.identity):
            layer.nonlinearity = nonlinearities.identity
            add_nonlinearity = True

        if add_bias:
            layer = BiasLayer(
                incoming=layer,
                b=b
            )
        if add_nonlinearity:
            layer = NonlinearityLayer(
                incoming=layer,
                nonlinearity=nonlinearity
            )
        incoming = layer
    return layer
