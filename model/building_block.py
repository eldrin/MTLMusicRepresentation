import numpy as np

from sklearn.externals import joblib

import theano
from lasagne import layers as L
from lasagne.nonlinearities import tanh, elu

from custom_layer import STFTLayer, MelSpecLayer, build_autoencoder, build_siamese
from utils.misc import get_in_shape

def conv2d(
    incoming, n_filters, filter_size, stride,
    pool_size, nonlinearity, batch_norm, name,
    verbose, *args, **kwargs):
    """"""
    if stride is None:
        stride=(1,1)

    layer = L.Conv2DLayer(
        incoming, num_filters=n_filters,
        filter_size=filter_size, stride=stride,
        pad='same', nonlinearity=None, name=name
    )
    if batch_norm:
        name += '.bn'
        layer = L.BatchNormLayer(layer, name=name)

    name += '.nonlin'
    layer = L.NonlinearityLayer(
        layer,
        nonlinearity=nonlinearity
    )
    return layer

def conv_block(
    net, n_convs, n_filters, filter_size,
    stride, pool_size, nonlinearity, batch_norm, name,
    prev_name, verbose=False, *args, **kwargs):
    """"""
    # check nonlin
    if isinstance(nonlinearity, (tuple, list)):
        if len(nonlinearity) != n_convs:
            raise ValueError(
                '[ERROR] if nonlinearity passed as Iterable of nonlins, \
                the length should be same as n_convs')
    else:
        nonlinearity = [nonlinearity] * n_convs

    # check stride
    if stride is None:
        stride = (1,1)

    # check filter_size
    if isinstance(filter_size, list):
        if n_convs != len(filter_size):
            raise ValueError(
                '[ERROR] number of filter sizes and n_conv should be equal!'
            )
    else:
        filter_size = [filter_size] * n_convs

    # add conv layers
    for n in xrange(n_convs):
        layer_name = name + '.{:d}'.format(n+1)
        if n == 0:
            prev = prev_name
        else:
            prev = next(reversed(net))

        net[layer_name] = L.Conv2DLayer(
            net[prev],
            num_filters=n_filters,
            filter_size=filter_size[n],
            stride=stride,
            pad='same',
            nonlinearity=None,
            name=layer_name
        )
        if batch_norm:
            layer_name += '.bn'
            net[layer_name] = L.BatchNormLayer(
                net[next(reversed(net))],
                name=layer_name
            )
        layer_name += '.nonlin'
        net[layer_name] = L.NonlinearityLayer(
            net[next(reversed(net))],
            nonlinearity=nonlinearity[n]
        )

    if pool_size != None:
        # add pooling layer
        layer_name = name + '.pl'
        net[layer_name] = L.MaxPool2DLayer(
            net[next(reversed(net))],
            pool_size=pool_size
        )

    if verbose:
        print(next(reversed(net)), net[next(reversed(net))].output_shape)

    return net


def input_block(net, config, melspec=False, verbose=True):
    """
    """
    # load scaler
    sclr = joblib.load(config.paths.preproc.scaler)

    net['input'] = L.InputLayer(
        shape=get_in_shape(config),
        name='input'
    )
    sigma = theano.shared(np.array(0.,dtype=np.float32), name='noise_controller')
    net['noise'] = L.GaussianNoiseLayer(
        net['input'], sigma=sigma, name='input_corruption')

    if config.hyper_parameters.input == "melspec":

        net['sclr'] = L.standardize(
            net['noise'],
            offset=sclr.mean_.astype(np.float32),
            scale=sclr.scale_.astype(np.float32),
            shared_axes=(0,1,2)
        )
    else:
        net['stft'] = STFTLayer(
            L.ReshapeLayer(
                net['noise'],
                ([0],[1],[2],1),
                name='reshape'
            ),
            n_fft=config.hyper_parameters.n_fft,
            hop_size=config.hyper_parameters.hop_size
        )

        if melspec:
            net['melspec'] = MelSpecLayer(
                sr=config.hyper_parameters.sample_rate,
                n_fft=config.hyper_parameters.n_fft,
                n_mels=128, log_amplitude=True)

            net['sclr'] = L.standardize(
                net['melspec'],
                offset=sclr.mean_.astype(np.float32),
                scale=sclr.scale_.astype(np.float32),
                shared_axes=(0,1,2)
            )

        else:
            net['sclr'] = L.standardize(
                net['stft'],
                offset=sclr.mean_.astype(np.float32),
                scale=sclr.scale_.astype(np.float32),
                shared_axes=(0,1,2)
            )

            # only pooling freq domain
            net['stft.pl'] = L.MaxPool2DLayer(
                net['sclr'],
                pool_size=(2,1),
                name='stft.pl'
            )

    if verbose:
        print(net['input'].output_shape)
        # if melspec:
        #     print(net['melspec'].output_shape)
        # else:
        #     print(net['stft'].output_shape)
        #     print(net['stft.pl'].output_shape)
        print(net['sclr'].output_shape)

    return net, sigma


def output_block(net, config, non_lin, verbose=True):
    """
    """
    # output setting
    out_acts = []
    for out_act in config.hyper_parameters.out_act:
        exec('from lasagne.nonlinearities import {}'.format(out_act))
        out_acts.append(eval(out_act))
    n_outs = config.hyper_parameters.n_out

    # Global Average Pooling
    last_conv_block_name = next(reversed(net))
    net['gap'] = L.GlobalPoolLayer(
        net[last_conv_block_name],
        name='gap'
    )
    net['gap.bn'] = L.BatchNormLayer(
        net['gap'],
        name='gap.bn'
    )
    n_features = net['gap.bn'].output_shape[-1]

    # feature Layer
    net['fc'] = L.dropout(
        L.batch_norm(
            L.DenseLayer(
                net['gap.bn'],
                num_units=n_features,
                nonlinearity=non_lin,
                name='fc'
            )
        ),
        name='fc.bn.do'
    )

    # output (prediction)
    # check whether the model if for MTL or STL
    # target is passed as list, regardless whether
    # it's MTL or STL (configuration checker checks it)
    targets = config.target
    out_layer_names = []
    for target, n_out, out_act in zip(targets, n_outs, out_acts):

        out_layer_names.append('out.{}'.format(target))

        if target == 'self':
            net[out_layer_names[-1]], inputs = build_siamese(net['fc'])
        else:
            net[out_layer_names[-1]] = L.DenseLayer(
                net['fc'],
                num_units=n_out,
                nonlinearity=out_act,
                name=out_layer_names[-1]
            )
            inputs = [net['input'].input_var]

    # make a concatation layer just for save/load purpose
    net['IO'] = L.ConcatLayer(
        [L.FlattenLayer(net[target_layer_name])
         if target == 'self' else net[target_layer_name]
         for target_layer_name in out_layer_names],
        name='IO'
    )

    if verbose:
        print(net['gap.bn'].output_shape)
        print(net['fc'].output_shape)
        for target in targets:
            print(net['out.{}'.format(target)].output_shape)

    return net, inputs

