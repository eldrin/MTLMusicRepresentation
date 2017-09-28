import numpy as np

from sklearn.externals import joblib

import theano
from lasagne import layers as L
from lasagne.nonlinearities import tanh, elu

from custom_layer import STFTLayer, build_autoencoder, build_siamese
from utils.misc import get_in_shape

def conv_block(
    net, n_convs, n_filters, filter_size,
    stride, pool_size, nonlinearity, batch_norm, name,
    is_feature=False, verbose=False, *args, **kwargs):
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
        net[layer_name] = L.Conv2DLayer(
            net[next(reversed(net))],
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
        if batch_norm:
            layer_name += '.bn'
            net[layer_name] = L.BatchNormLayer(
                net[next(reversed(net))],
                name=layer_name
            )

    if verbose:
        print(next(reversed(net)), net[next(reversed(net))].output_shape)

    return net


def input_block(net, config, verbose=True):
    """
    """
    # load scaler
    sclr = joblib.load(config.paths.preproc.scaler)

    # prepare input layer
    net['input'] = L.InputLayer(
        shape=get_in_shape(config),
        name='input'
    )

    sigma = theano.shared(np.array(0.,dtype=np.float32), name='noise_controller')
    net['noise'] = L.GaussianNoiseLayer(
        net['input'], sigma=sigma, name='input_corruption')

    net['stft'] = STFTLayer(
        L.ReshapeLayer(
            net['noise'],
            ([0],[1],[2],1),
            name='reshape'
        ),
        n_fft=config.hyper_parameters.n_fft,
        hop_size=config.hyper_parameters.hop_size
    )

    net['sclr'] = L.standardize(
        net['stft'],
        offset=sclr.mean_.astype(np.float32),
        scale=sclr.scale_.astype(np.float32),
        shared_axes=(0,1,3)
    )
    # only pooling freq domain
    net['stft.pl'] = L.batch_norm(
        L.MaxPool2DLayer(
            net['sclr'],
            pool_size=(2,1),
            name='stft.pl'
        )
    )

    if verbose:
        print(net['input'].output_shape)
        print(net['stft'].output_shape)
        print(net['stft.pl'].output_shape)

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
            """
            layers, net['fc'] = build_autoencoder(
                net['fc'], nonlinearity=tanh, learnable_conv=False)

            # # STFT (sclr) reconstruction
            # net[out_layer_names[-1]] = layers[-5]

            # STFT (sclr) after couple of  trainable 2DConvs
            layer = L.Conv2DLayer(
                layers[-5], num_filters=32, pad='same', filter_size=(3, 3),
                nonlinearity=tanh, name='AEout1')
            layer = L.Conv2DLayer(
                layer, num_filters=32, pad='same', filter_size=(3, 3),
                nonlinearity=tanh, name='AEout2')

            net[out_layer_names[-1]] = L.Conv2DLayer(
                layer, num_filters=2, pad='same', filter_size=(3, 3),
                nonlinearity=None, name='AEout_out')

            # # Signal reconstruction
            # net[out_layer_names[-1]] = layers[-1]
            """
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

