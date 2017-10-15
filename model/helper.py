import os
import numpy as np
from sklearn.externals import joblib

import namedtupled

import theano
import lasagne
from theano import tensor as T

from lasagne import layers as L
from lasagne.nonlinearities import softmax, linear
from lasagne.regularization import regularize_layer_params, l2
from lasagne.updates import (sgd,
                             nesterov_momentum,
                             adagrad,
                             adam,
                             adamax,
                             rmsprop,
                             adadelta)

from utils.misc import get_layer
from ops import mu_law_encode

floatX = theano.config.floatX


def categorical_crossentropy_over_signal(pred, true):
    """ Might be better than l2 norm on signal,
    but extremely memory expensive
    """
    true = mu_law_encode(true)
    return T.nnet.categorical_crossentropy(pred, true)


def categorical_crossentropy_over_spectrum(pred, true):
    # normalize in frequency side
    true_n = true / (true.sum(axis=2)[:, :, None, :])
    true_n = true_n.dimshuffle(0, 1, 3, 2)
    true_n = true_n.reshape((true_n.shape[0] * true_n.shape[1] *
                             true_n.shape[2], true_n.shape[-1]))

    pred = pred.dimshuffle(0, 1, 3, 2)
    pred = pred.reshape((pred.shape[0] * pred.shape[1] *
                         pred.shape[2], pred.shape[-1]))

    pred = T.nnet.softmax(pred)

    return T.nnet.nnet.categorical_crossentropy(pred, true_n)


def binary_crossentropy_over_spectrum(pred, true):
    # normalize in frequency side
    true_n = true / (true.sum(axis=2)[:, :, None, :] + 0.000001)
    pred = T.nnet.sigmoid(pred)
    return T.nnet.nnet.binary_crossentropy(pred, true_n)


def get_train_funcs(net, config, input_vars=None, **kwargs):
    """
    """
    # optimizer setting
    optimizer = eval(config.hyper_parameters.optimizer)
    lr = config.hyper_parameters.learning_rate
    beta = config.hyper_parameters.l2

    # currently, we will only use 'fc' output as feature
    if config.hyper_parameters.branch_at != 'fc':
        feature_layer = '{}.fc'
    else:
        feature_layer = 'fc'

    # function containor
    functions = {}

    for target in config.target:
        # get access point
        out_layer_name = '{}.out'.format(target)
        input_name = '{}.inputs'.format(target)
        input_var = input_vars[input_name]

        layers = L.get_all_layers(net[out_layer_name])

        O = L.get_output(net[out_layer_name], deterministic=False)
        O_vl = L.get_output(net[out_layer_name], deterministic=True)

        train_params = L.get_all_params(net[out_layer_name], trainable=True)

        if target == 'self':
            train_params.extend(
                L.get_all_params(net['self.fc'], trainable=True))

        if net[out_layer_name].nonlinearity == softmax:
            loss = lasagne.objectives.categorical_crossentropy
        elif net[out_layer_name].nonlinearity == linear:
            loss = lasagne.objectives.squared_error
        Y = T.matrix('target')

        error = loss(O, Y).mean()
        error_vl = loss(O_vl, Y).mean()

        # regularization terms
        reg_DNN = beta * regularize_layer_params(layers, l2)

        cost = error + reg_DNN
        cost_vl = error_vl + reg_DNN

        # prepare update rule
        updates = optimizer(cost, train_params, learning_rate=lr)

        # compile functions
        if target == 'self':
            # cost_rel_inputs = [layers[0].input_var,
            #                    target_net['input'].input_var]
            if input_var is not None:
                cost_rel_inputs = list(input_var)
                cost_rel_inputs.append(Y)
                input_var_feat = [cost_rel_inputs[0]]
                input_var_pred = input_var
            else:
                raise ValueError(
                    '[ERROR] In "self" case, you need to pass input vars')
        else:
            if input_var is not None:
                cost_rel_inputs = list(input_var)
                cost_rel_inputs.append(Y)
            else:
                cost_rel_inputs = [layers[0].input_var, Y]
            input_var_feat = [layers[0].input_var]
            input_var_pred = input_var_feat

        # feature is also class dependent
        feature = L.get_output(get_layer(net, feature_layer.format(target)),
                               inputs=input_var_feat[0], deterministic=True)

        functions[target] = {}
        functions[target]['train'] = theano.function(
            inputs=cost_rel_inputs,
            updates=updates,
            allow_input_downcast=True
        )
        functions[target]['valid'] = theano.function(
            inputs=cost_rel_inputs,
            outputs=cost_vl,
            allow_input_downcast=True
        )
        functions[target]['predict'] = theano.function(
            inputs=input_var_pred,
            outputs=O_vl,
            allow_input_downcast=True
        )
        functions[target]['feature'] = theano.function(
            inputs=input_var_feat,
            outputs=feature,
            allow_input_downcast=True
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
        feature_layer, deterministic=True)
    functions['features'] = theano.function(
        inputs=[layers[0].input_var],
        outputs=feature,
        allow_input_downcast=True
    )

    # implement feature map per each layer
    features = [
        L.get_output(
            get_layer(net, layer), deterministic=True
        )
        for layer
        in ['conv1.pl', 'conv2.pl', 'conv3.pl', 'conv4.pl']
    ]
    functions['filter_response'] = theano.function(
        inputs=[layers[0].input_var],
        outputs=features,
        allow_input_downcast=True
    )

    # prediction / CAM are output dependant
    for target in config.target:
        out_layer_name = '{}.out'.format(target)
        functions[target] = {}

        c = T.iscalar('class_idx')
        i = T.iscalar('item_idx')

        output_layer = get_layer(net, out_layer_name)

        O_vl, CAM_vl = L.get_output(
            [output_layer, cam_layer], deterministic=True)

        # prediction function
        functions[target]['predict'] = theano.function(
            inputs=[layers[0].input_var],
            outputs=O_vl,
            allow_input_downcast=True
        )

        # implement grad-cam
        filter_weight = T.grad(O_vl[i, c], wrt=[CAM_vl])[0]
        filter_weight_gap = filter_weight.mean(axis=(2, 3))

        weighted_feature_map = T.nnet.relu(
            (filter_weight_gap[:, :, None, None] * CAM_vl).sum(axis=1)
        )

        functions[target]['gradcam'] = theano.function(
            inputs=[layers[0].input_var, i, c],
            outputs=weighted_feature_map,
            allow_input_downcast=True
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


# def test_model_building():
#     """
#     """
#     # model_comp = build_convae_deep(
#     #     in_shape=(None,2,22050 * 5 - 22050 * 5 % 256)
#     # )
#     model_comp = build_2dconv_clf_deep(
#         in_shape=(None,2,513,432)
#     )


def save_check_point(it, network, config):
    """
    """
    fns = get_check_point_fns(config)
    config_dict = namedtupled.reduce(config)

    np.savez(fns['param'],
             *lasagne.layers.get_all_param_values(network.values()))
    joblib.dump({'iter': it, 'config': config_dict}, fns['state'])


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

            lasagne.layers.set_all_param_values(network.values(),
                                                param_values)
            it = joblib.load(fns['state'])['iter']
        except Exception as e:
            print(e)
            print('Cannot load parameters!')
    else:
        print('Cannot find parameters!')

    return it, network


def get_check_point_fns(config):
    """"""
    fns = {}
    fns['param'] = None
    fns['state'] = None

    dump_root = config.paths.model
    target_string = '_'.join(config.target)
    fname = config.paths.file_name.format(target_string)
    suffix_param = '_param.npz'
    suffix_state = '_state.dat.gz'

    try:
        fns['param'] = os.path.join(dump_root, fname + suffix_param)
        fns['state'] = os.path.join(dump_root, fname + suffix_state)

    except Exception as e:
        raise e  # TODO: elaborate this

    return fns


if __name__ == "__main__":
    pass
    # test_model_building()
