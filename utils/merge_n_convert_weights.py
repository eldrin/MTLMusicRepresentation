import os
from collections import OrderedDict
from functools import partial
import subprocess
# import inspect

import theano
import namedtupled
import numpy as np

from sklearn.externals import joblib
from lasagne import layers as L
from lasagne.nonlinearities import rectify, softmax

from model.model import Model

import fire

PATCH_SCRIPT_FN = '/mnt/bulk2/exp_res/utils/patch_state.py'

ARCH_TEMPLATE = [
    # input block
    'input', 'noise', 'sclr',
    # conv-pool block
    'conv1.1', 'conv1.1.bn', 'conv1.1.bn.nonlin', 'conv1.pl',
    'conv2.1', 'conv2.1.bn', 'conv2.1.bn.nonlin', 'conv2.pl',
    'conv3.1', 'conv3.1.bn', 'conv3.1.bn.nonlin', 'conv3.pl',
    'conv4.1', 'conv4.1.bn', 'conv4.1.bn.nonlin', 'conv4.pl',
    'conv5.1', 'conv5.1.bn', 'conv5.1.bn.nonlin', 'conv5.pl',
    'conv6.1', 'conv6.1.bn', 'conv6.1.bn.nonlin',
    'conv6.2', 'conv6.2.bn', 'conv6.2.bn.nonlin',
    'gap_bn',
    # output block
    'fc', 'fc_bn', 'fc_bn_nonlin', 'fc_dropout',
    'out'
]

NET_TEMPLATE = OrderedDict([
    ('input', (L.InputLayer, {'shape': (None, 2, 216, 128)})),
    ('noise', (L.GaussianNoiseLayer, {'sigma': 0.})),
    ('sclr', (L.standardize, {'shared_axes': (0, 1, 2)})),
    ('conv1.1', (L.Conv2DLayer,
                 {'stride': (2, 1), 'pad': 'same', 'nonlinearity': None})),
    ('conv1.1.bn', (L.BatchNormLayer, None)),
    ('conv1.1.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('conv1.1.pl', (L.MaxPool2DLayer, {'pool_size': (2, 2)})),
    ('conv2.1', (L.Conv2DLayer, {'pad': 'same', 'nonlinearity': None})),
    ('conv2.1.bn', (L.BatchNormLayer, None)),
    ('conv2.1.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('conv2.1.pl', (L.MaxPool2DLayer, {'pool_size': (2, 2)})),

    ('conv3.1', (L.Conv2DLayer, {'pad': 'same', 'nonlinearity': None})),
    ('conv3.1.bn', (L.BatchNormLayer, None)),
    ('conv3.1.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('conv3.1.pl', (L.MaxPool2DLayer, {'pool_size': (2, 2)})),

    ('conv4.1', (L.Conv2DLayer, {'pad': 'same', 'nonlinearity': None})),
    ('conv4.1.bn', (L.BatchNormLayer, None)),
    ('conv4.1.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('conv4.1.pl', (L.MaxPool2DLayer, {'pool_size': (2, 2)})),

    ('conv5.1', (L.Conv2DLayer, {'pad': 'same', 'nonlinearity': None})),
    ('conv5.1.bn', (L.BatchNormLayer, None)),
    ('conv5.1.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('conv5.1.pl', (L.MaxPool2DLayer, {'pool_size': (2, 2)})),

    ('conv6.1', (L.Conv2DLayer, {'pad': 'same', 'nonlinearity': None})),
    ('conv6.1.bn', (L.BatchNormLayer, None)),
    ('conv6.1.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('conv6.2', (L.Conv2DLayer, {'pad': 'same', 'nonlinearity': None})),
    ('conv6.2.bn', (L.BatchNormLayer, None)),
    ('conv6.2.bn.nonlin', (L.NonlinearityLayer, {'nonlinearity':
                                                 rectify})),
    ('gap', (L.GlobalPoolLayer, None)),
    ('gap_bn', (L.BatchNormLayer, None)),
    ('fc', (L.DenseLayer, {'nonlinearity': None, 'b': None})),
    ('fc_bn', (L.BatchNormLayer, None)),
    ('fc_bn_nonlin', (L.NonlinearityLayer, {'nonlinearity': rectify})),
    ('fc_dropout', (L.DropoutLayer, None)),
    ('out', (L.DenseLayer, {'nonlinearity': softmax}))
])


def find_layer(l, kw):
    """ find layer with keyword """
    if l.name is not None:
        return kw in l.name
    else:
        return False


def _merge_parameters(layers, cur_layer, n_lrn_src,
                      branch_at, verbose=False):
    """"""
    # resulting containor
    param_res = {}

    # fetch saved layers
    condition = partial(find_layer, kw=cur_layer)
    with_layer_n = filter(condition, layers)
    if 'bn' not in cur_layer:
        with_layer_n = filter(lambda l: 'bn' not in l.name, with_layer_n)

    # filter layers without weights
    with_layer_n = filter(lambda l: len(l.params) > 0, with_layer_n)
    if len(with_layer_n) == 1:
        n_lrn_src = 1

    # start process
    if len(with_layer_n) > 0:
        if 'bn' not in cur_layer:
            if 'conv' in cur_layer:  # Conv2DLayer

                # merge weight
                if str(branch_at) + '.1' in cur_layer:  # do the out-dim merge
                    W = np.concatenate(
                        map(lambda l: l.W.get_value(), with_layer_n),
                        axis=0
                    )
                else:  # do the full merge (diagonal block tensor)
                    o_, i_, w_, h_ = with_layer_n[0].W.get_value().shape
                    W = np.zeros([o_ * n_lrn_src, i_ * n_lrn_src, w_, h_])
                    for i, l in enumerate(with_layer_n):
                        slc_0 = slice(i * o_, (i+1) * o_)  # out ch dim
                        slc_1 = slice(i * i_, (i+1) * i_)  # in ch dim
                        W[slc_0, slc_1] = l.W.get_value()

                # merge bias
                b = np.concatenate(
                    map(lambda l: l.b.get_value(), with_layer_n)
                )

            else:  # DenseLayer

                # merge weight
                if branch_at == 'fc':  # do the out-dim merge
                    W = np.concatenate(
                        map(lambda l: l.W.get_value(), with_layer_n),
                        axis=1
                    )
                else:  # do the full merge (diagonal block matrix)
                    i_, o_ = with_layer_n[0].W.get_value().shape
                    W = np.zeros([i_ * n_lrn_src, o_ * n_lrn_src])
                    for i, l in enumerate(with_layer_n):
                        slc_0 = slice(i * i_, (i+1) * i_)  # out ch dim
                        slc_1 = slice(i * o_, (i+1) * o_)  # in ch dim
                        W[slc_0, slc_1] = l.W.get_value()

                # merge bias
                if with_layer_n[0].b is not None:
                    o_, = with_layer_n[0].b.get_value().shape
                    b = np.zeros([o_ * n_lrn_src, ])
                    for i, l in enumerate(with_layer_n):
                        b[i * o_: (i+1) * o_] = l.b.get_value()
                else:
                    b = None

            if verbose:
                print cur_layer, W.shape, b.shape if b is not None else None
            param_res['W'] = W
            param_res['b'] = b

        elif 'bn' in cur_layer:
            gamma = np.concatenate(
                map(lambda l: l.gamma.get_value(), with_layer_n))
            beta = np.concatenate(
                map(lambda l: l.beta.get_value(), with_layer_n))
            mean = np.concatenate(
                map(lambda l: l.mean.get_value(), with_layer_n))
            inv_std = np.concatenate(
                map(lambda l: l.inv_std.get_value(), with_layer_n))

            if verbose:
                print cur_layer, gamma.shape, \
                        beta.shape, mean.shape, inv_std.shape
            param_res['gamma'] = gamma
            param_res['beta'] = beta
            param_res['mean'] = mean
            param_res['inv_std'] = inv_std

        else:
            print with_layer_n

        return param_res


def fetch_merged_parameters(model):
    """"""
    global ARCH_TEMPLATE
    # get all layers
    all_layers = L.get_all_layers(model.net.values())

    # set some variables
    lrn_src = model.config.target
    n_lrn_src = len(lrn_src)
    branch_at = model.config.hyper_parameters.branch_at

    # start process
    weights = OrderedDict()

    # following latyers manually processed (always shared blocks)
    weights['sclr'] = {}
    weights['sclr']['offset'] = -all_layers[2].b.get_value()
    weights['sclr']['scale'] = 1./all_layers[3].params.keys()[0].get_value()
    # print 'sclr', weights['sclr']['offset'].shape,\
    #     weights['sclr']['scale'].shape

    for layer in ARCH_TEMPLATE:
        if layer in ['input', 'noise', 'sclr']:
            continue
        weight = _merge_parameters(all_layers, layer, n_lrn_src, branch_at)
        if weight is not None:
            weights[layer] = weight

    return weights


def init_layers(layer_list, weights, output=False):
    """"""
    # update parameters
    for k, v in weights.iteritems():
        if layer_list[k][1] is not None:
            if layer_list[k][0] == L.Conv2DLayer:
                v.update(
                    {'num_filters': v['W'].shape[0],
                     'filter_size': v['W'].shape[2:]}
                )
            if layer_list[k][0] == L.DenseLayer:
                v.update({'num_units': v['W'].shape[1]})
            layer_list[k][1].update(v)
        else:
            layer_list[k] = (layer_list[k][0], v)

    # initialize layers
    for name, layer_param in layer_list.iteritems():
        if not output:
            if name == 'out':
                break

            if name == 'input':
                net = layer_param[0](name=name, **layer_param[1])
            else:
                if name != 'sclr':
                    if layer_param[1] is not None:
                        net = layer_param[0](net, name=name, **layer_param[1])
                    else:
                        net = layer_param[0](net, name=name)
                else:
                    net = layer_param[0](net, **layer_param[1])
                # print name, net.output_shape

    return net


def merge_branch_net(model_state_fn):
    """"""
    global NET_TEMPLATE
    state = joblib.load(model_state_fn)
    model = Model(namedtupled.map(state).config)

    w = fetch_merged_parameters(model)
    net = init_layers(NET_TEMPLATE, w)
    return net, model


def save_model(net, path):
    """ TODO """
    params = OrderedDict(
        [(param.name, param.get_value())
         for param in L.get_all_params(net)]
    )
    np.savez(path, **params)


def test(model, net):
    """"""
    # merge network
    branch_at = model.config.hyper_parameters.branch_at
    targets = model.config.target

    # compile feature function
    feature = L.get_output(net, deterministic=True)
    layers_rebuild = L.get_all_layers(net)
    f_feature = theano.function([layers_rebuild[0].input_var],
                                feature, allow_input_downcast=True)

    # test with random init
    x = np.random.rand(1, 2, 216, 128)
    z_hat = f_feature(x)
    if branch_at == 'fc':
        z = model.feature(model.config.target[0], x)
    else:
        z = np.concatenate([model.feature(t, x).ravel() for t in targets])

    print 'all same?:', np.isclose(z_hat, z).all()
    print 'how same?: {:.2f}'.format(
        np.isclose(z_hat, z).sum() / float(np.prod(z.shape)))


def branch2single_net(model_state_fn, output_path):
    """"""
    net, model = merge_branch_net(model_state_fn)
    test(model, net)
    save_model(net, output_path)


def process_all(root):
    """"""
    global PATCH_SCRIPT_FN
    for root, dirs, files in os.walk(root):
        if len(dirs) == 0:
            if len(filter(lambda l: 'singlenet' in l, files)) > 0:
                continue

            fn = os.path.join(
                root, filter(lambda l: 'state' in l, files)[0]
            )
            out_fn = '_'.join(fn.split('_')[:-1]) + '_param_singlenet.npz'

            # just in case
            subprocess.call(['python', PATCH_SCRIPT_FN, fn])
            # process
            branch2single_net(fn, out_fn)


if __name__ == "__main__":
    # fire.Fire(test)
    # fire.Fire(branch2single_net)
    fire.Fire(process_all)
