import os
import numpy as np
from sklearn.externals import joblib

import theano
import lasagne
from theano import tensor as T

floatX = theano.config.floatX
from lasagne import layers as L
from lasagne.nonlinearities import rectify, tanh, softmax, elu, linear
from lasagne.regularization import regularize_layer_params, l2, l1
from lasagne.updates import sgd,nesterov_momentum,adagrad,adam,rmsprop,adamax

from utils.misc import *

def get_train_funcs(net, config, feature_layer=None, **kwargs):
    """
    """
    # optimizer setting
    optimizer = eval(config.hyper_parameters.optimizer)
    lr = config.hyper_parameters.learning_rate
    beta = config.hyper_parameters.l2

    if feature_layer is None:
        feature_layer = 'fc.bn.do'

    # function containor
    functions = {}

    for target in config.target:
        # get access point
        out_layer_name = 'out.{}'.format(target)

        layers = L.get_all_layers(net[out_layer_name])

        if net[out_layer_name].nonlinearity == softmax:
            loss = lasagne.objectives.categorical_crossentropy
        elif net[out_layer_name].nonlinearity == linear:
            loss = lasagne.objectives.squared_error

        Y = T.matrix('target')

        O = L.get_output(net[out_layer_name],deterministic=False)
        O_vl = L.get_output(net[out_layer_name],deterministic=True)

        train_params = L.get_all_params(net[out_layer_name],trainable=True)
        non_train_params = L.get_all_params(net[out_layer_name],trainable=False)

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
        functions[target] = {}

        functions[target]['train'] = theano.function(
            inputs = cost_rel_inputs,
            updates = updates,
            allow_input_downcast=True
        )
        functions[target]['valid'] = theano.function(
            inputs = cost_rel_inputs,
            outputs = cost_vl,
            allow_input_downcast=True
        )
        functions[target]['predict'] = theano.function(
            inputs = [layers[0].input_var],
            outputs = O_vl,
            allow_input_downcast = True
        )

    # feature is class independent
    feature = L.get_output(net[feature_layer], deterministic=True)
    functions['feature'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = feature,
        allow_input_downcast = True
    )

    return functions


def get_debug_funcs(net, feature_layer, cam_layer, config):
    """
    """

    layers = L.get_all_layers(net)
    cam_layer = get_layer(net, cam_layer)
    feature_layer = get_layer(net, feature_layer)

    # function containor
    functions = {}

    # feature is not output dependant
    feature = L.get_output(
        feature_layer,deterministic=True)
    functions['features'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = feature,
        allow_input_downcast = True
    )

    # implement feature map per each layer
    features = [
        L.get_output(
            get_layer(net, layer), deterministic=True
        )
        for layer
        in ['conv1.pl','conv2.pl','conv3.pl','conv4.pl']
    ]
    functions['filter_response'] = theano.function(
        inputs = [layers[0].input_var],
        outputs = features,
        allow_input_downcast = True
    )

    # prediction / CAM are output dependant
    for target in config.target:
        out_layer_name = 'out.{}'.format(target)
        functions[target] = {}

        c = T.iscalar('class_idx')
        i = T.iscalar('item_idx')

        output_layer = get_layer(net, out_layer_name)

        O_vl, CAM_vl = L.get_output(
            [output_layer,cam_layer], deterministic=True)

        # prediction function
        functions[target]['predict'] = theano.function(
            inputs = [layers[0].input_var],
            outputs = O_vl,
            allow_input_downcast = True
        )

        # implement grad-cam
        filter_weight = T.grad(O_vl[i,c],wrt=[CAM_vl])[0]
        filter_weight_gap = filter_weight.mean(axis=(2,3))

        weighted_feature_map = T.nnet.relu(
            (filter_weight_gap[:,:,None,None] * CAM_vl).sum(axis=1)
        )

        functions[target]['gradcam'] = theano.function(
            inputs = [layers[0].input_var,i,c],
            outputs = weighted_feature_map,
            allow_input_downcast = True
        )

        # TODO: saliency map is not bit treaky for MTL case
        #       figure out how to build generalizable codes

        # # implement saliency map
        # relu_layers = [
        #     layer for layer in layers
        #     if getattr(layer, 'nonlinearity', None) is rectify
        # ]
        # modded_relu = GuidedBackprop(rectify)
        # for layer in relu_layers:
        #     layer.nonlinearity = modded_relu

        # O_vl = L.get_output(net, deterministic=True)

        # saliency = T.grad(O_vl[i,c],wrt=[layers[0].input_var])[0]

        # functions['saliency'] = theano.function(
        #     inputs = [layers[0].input_var,i,c],
        #     outputs = saliency,
        #     allow_input_downcast = True
        # )

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

            lasagne.layers.set_all_param_values(network['IO'], param_values)
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

    np.savez(fns['param'], *lasagne.layers.get_all_param_values(network['IO']))
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


if __name__ == "__main__":
    test_model_building()
