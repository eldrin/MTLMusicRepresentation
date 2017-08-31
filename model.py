import os
import sys

from functools import partial
import gzip
import cPickle as pkl
import h5py
from collections import OrderedDict

import numpy as np
import pandas as pd

import theano
import lasagne
from theano import sparse as tsp
from theano import tensor as T

floatX = theano.config.floatX
from lasagne import layers as L
from lasagne.nonlinearities import rectify, tanh, softmax, elu, linear
from lasagne.regularization import regularize_layer_params, l2, l1
from lasagne.updates import sgd,nesterov_momentum,adagrad,adam,rmsprop,adamax

from utils import *
from custom_layers import STFTLayer

from sklearn.externals import joblib
import scipy

def build_2dconv_clf_deep(config, **kwargs):
    """
    """

    # load scaler
    sclr = joblib.load(config.paths.preproc.scaler)

    # activation setting
    exec(
        "from lasagne.nonlinearities import {}".format(
        config.hyper_parameters.activation)
    )
    non_lin = eval(config.hyper_parameters.activation)

    # output setting
    exec('from lasagne.nonlinearities import {} as out_act'.format(
        config.hyper_parameters.out_act)
    )
    n_out = config.hyper_parameters.n_out

    # network dict
    net = {}

    # prepare input layer
    net['input'] = L.InputLayer(
        shape=get_in_shape(config),
        name='input'
    )
    print(net['input'].output_shape)

    net['stft'] = STFTLayer(
        L.ReshapeLayer(
            net['input'],
            ([0],[1],[2],1),
            name='reshape'
        ),
        n_fft=config.hyper_parameters.n_fft,
        hop_size=config.hyper_parameters.hop_size
    )
    print(net['stft'].output_shape)

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
    print(net['stft.pl'].output_shape)

    net['conv1'] = L.batch_norm(
        L.Conv2DLayer(
            net['stft.pl'],
            num_filters=64,
            filter_size=(5,5),
            stride=(2,2),
            pad='same',
            nonlinearity=non_lin,
            name='conv1'
        )
    )
    net['conv1.pl'] = L.batch_norm(
        L.MaxPool2DLayer(
            net['conv1'],
            pool_size=(2,2),
            name='conv1.pl'
        )
    )
    print(net['conv1.pl'].output_shape)

    net['conv21'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv1.pl'],
            num_filters=128,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv21'
        )
    )
    net['conv2.pl'] = L.batch_norm(
        L.MaxPool2DLayer(
            net['conv21'],
            pool_size=(2,2),
            name='conv2.pl'
        )
    )
    print(net['conv2.pl'].output_shape)

    net['conv31'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv2.pl'],
            num_filters=256,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv31'
        )
    )
    net['conv32'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv31'],
            num_filters=256,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv32'
        )
    )
    net['conv3.pl'] = L.batch_norm(
        L.MaxPool2DLayer(
            net['conv32'],
            pool_size=(2,2),
            name='conv3.pl'
        )
    )
    print(net['conv3.pl'].output_shape)

    net['conv41'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv3.pl'],
            num_filters=384,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv41'
        )
    )
    net['conv42'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv41'],
            num_filters=384,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv42'
        )
    )
    net['conv43'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv42'],
            num_filters=384,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv43'
        )
    )
    net['conv4.pl'] = L.batch_norm(
        L.MaxPool2DLayer(
            net['conv43'],
            pool_size=(2,2),
            name='conv4.pl'
        )
    )
    print(net['conv4.pl'].output_shape)

    net['conv51'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv4.pl'],
            num_filters=512,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv51'
        )
    )
    net['conv52'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv51'],
            num_filters=512,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv52'
        )
    )
    net['conv53'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv52'],
            num_filters=512,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv53'
        )
    )
    net['conv5.pl'] = L.batch_norm(
        L.MaxPool2DLayer(
            net['conv53'],
            pool_size=(2,2),
            name='conv5.pl'
        )
    )
    print(net['conv5.pl'].output_shape)

    net['conv61'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv5.pl'],
            num_filters=512,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv61'
        )
    )
    net['conv62'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv61'],
            num_filters=512,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv62'
        )
    )
    net['conv63'] = L.batch_norm(
        L.Conv2DLayer(
            net['conv62'],
            num_filters=512,
            filter_size=(3,3),
            pad='same',
            nonlinearity=non_lin,
            name='conv63'
        )
    )
    print(net['conv63'].output_shape)

    net['gap'] = L.GlobalPoolLayer(
        net['conv63'],
        name='gap'
    )
    net['gap.bn'] = L.BatchNormLayer(
        net['gap'],
        name='gap.bn'
    )
    print(net['gap.bn'].output_shape)

    net['fc'] = L.dropout(
        L.batch_norm(
            L.DenseLayer(
                net['gap.bn'],
                num_units=512,
                nonlinearity=non_lin,
                name='fc'
            )
        ),
        name='fc.bn.do'
    )
    print(net['fc'].output_shape)

    net['out'] = L.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=out_act,
        name='output'
    )
    print(net['out'].output_shape)

    n_params = {}
    for layer in L.get_all_layers(net['out']):
        n_params[layer.name] = L.count_params(layer)

    print(sum(n_params.values()))

    return {'out':net['out'],'cam':net['conv61']}


def get_clf_model(net, config, **kwargs):
    """
    """

    # optimizer setting
    exec(
        "from lasagne.updates import {}".format(
        config.hyper_parameters.optimizer)
    )
    optimizer = eval(config.hyper_parameters.optimizer)
    lr = config.hyper_parameters.learning_rate
    beta = config.hyper_parameters.l2

    layers = L.get_all_layers(net['out'])

    if net['out'].nonlinearity == softmax:
        loss = lasagne.objectives.categorical_crossentropy
    elif net['out'].nonlinearity == linear:
        loss = lasagne.objectives.squared_error

    Y = T.matrix('target')

    O = L.get_output(net['out'],deterministic=False)
    O_vl,CAM_vl = L.get_output([net['out'],net['cam']],deterministic=True)

    train_params = L.get_all_params(net['out'],trainable=True)
    non_train_params = L.get_all_params(net['out'],trainable=False)

    error = loss(O,Y).mean()
    error_vl = loss(O_vl,Y).mean()

    # regularization terms
    reg_DNN = beta * regularize_layer_params(layers,l2)

    cost = error + reg_DNN
    cost_vl = error_vl + reg_DNN

    # prepare update rule
    updates = optimizer(cost,train_params,learning_rate=lr)

    # compile functions
    cost_rel_inputs = [layers[0].input_var,Y]

    functions = {}

    functions['train'] = theano.function(
        inputs = cost_rel_inputs,
        updates = updates,
        allow_input_downcast=True
    )

    functions['valid'] = theano.function(
        inputs = cost_rel_inputs,
        outputs = cost_vl,
        allow_input_downcast=True
    )

    functions['predict'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = O_vl,
        allow_input_downcast = True
    )

    return functions,train_params,non_train_params,updates


def get_debug_funcs(net, config):
    """
    """

    layers = L.get_all_layers(net['out'])

    Y = T.matrix('target')
    c = T.iscalar('class_idx')
    i = T.iscalar('item_idx')

    O_vl,CAM_vl = L.get_output([net['out'],net['cam']],deterministic=True)

    # cost funcgtion
    error_vl = lasagne.objectives.categorical_crossentropy(
        O_vl,Y
    ).mean()

    reg_DNN = regularize_layer_params(layers,l2)
    reg_DNN *= config.hyper_parameters.l2

    cost_vl = error_vl + reg_DNN

    functions = {}

    # prediction function
    functions['predict'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = O_vl,
        allow_input_downcast = True
    )

    # validation function
    functions['valid'] = theano.function(
        inputs = [layers[0].input_var,Y],
        outputs = cost_vl,
        allow_input_downcast=True
    )

    # GAP feature function
    features_gap = [
        L.get_output(
            L.GlobalPoolLayer(
                filter(lambda x:x.name==layer,layers)[0]
            ),
            deterministic=True
        )
        for layer
        in ['conv1.pl','conv2.pl','conv3.pl','conv4.pl']
    ]

    features_gap.extend(
        [
            L.get_output(
                filter(lambda x:x.name==layer,layers)[0],
                deterministic=True
            )
            for layer in ['gap.bn','fc.bn.do']
        ]
    )

    functions['features'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = features_gap,
        allow_input_downcast = True
    )

    # implement feature map per each layer
    features = [
        L.get_output(
            filter(lambda x:x.name==layer,layers)[0],
            deterministic=True
        )
        for layer
        in ['conv1.pl','conv2.pl','conv3.pl','conv4.pl']
    ]

    functions['filter_response'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = features,
        allow_input_downcast = True
    )

    # implement grad-cam
    filter_weight = T.grad(O_vl[i,c],wrt=[CAM_vl])[0]
    filter_weight_gap = filter_weight.mean(axis=(2,3))

    weighted_feature_map = T.nnet.relu(
        (filter_weight_gap[:,:,None,None] * CAM_vl).sum(axis=1)
    )

    functions['gradcam'] = theano.function(
        inputs = [layers[0].input_var,i,c],
        outputs = weighted_feature_map,
        allow_input_downcast = True
    )

    # implement saliency map
    relu_layers = [
        layer for layer in layers
        if getattr(layer, 'nonlinearity', None) is rectify
    ]
    modded_relu = GuidedBackprop(rectify)
    for layer in relu_layers:
        layer.nonlinearity = modded_relu

    O_vl = L.get_output(net['out'],deterministic=True)

    saliency = T.grad(O_vl[i,c],wrt=[layers[0].input_var])[0]

    functions['saliency'] = theano.function(
        inputs = [layers[0].input_var,i,c],
        outputs = saliency,
        allow_input_downcast = True
    )

    return functions


def test_model_building():
    """
    """
    # model_comp = build_convae_deep(
    #     in_shape=(None,2,22050 * 5 - 22050 * 5 % 256)
    # )
    model_comp = build_2dconv_clf_deep(
        in_shape=(None,2,513,432)
    )


if __name__ == "__main__":

    test_model_building()
