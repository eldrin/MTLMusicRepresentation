from collections import OrderedDict

from lasagne import layers as L
from building_block import input_block, conv_block, output_block

def build_2dconv_clf_deep(config, **kwargs):
    """
    """
    # activation setting
    exec(
        "from lasagne.nonlinearities import {}".format(
        config.hyper_parameters.activation)
    )
    non_lin = eval(config.hyper_parameters.activation)

    # network dict
    net = OrderedDict()
    net, sigma = input_block(net, config)
    net = conv_block(
        net, n_convs=1, n_filters=64, filter_size=(5,5),
        stride=(2,2), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv1', verbose=True
    )
    net = conv_block(
        net, n_convs=1, n_filters=128, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv2', verbose=True
    )
    net = conv_block(
        net, n_convs=2, n_filters=256, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv3', verbose=True
    )
    net = conv_block(
        net, n_convs=3, n_filters=384, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv4', verbose=True
    )
    net = conv_block(
        net, n_convs=3, n_filters=512, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv5', verbose=True
    )
    net = conv_block(
        net, n_convs=3, n_filters=512, filter_size=[(3,3),(3,3),(1,1)],
        stride=(1,1), pool_size=None, nonlinearity=non_lin,
        batch_norm=True, name='conv6', verbose=True
    )
    net = output_block(net, config, non_lin)

    # n_params = {}
    # for layer in L.get_all_layers(net['out']):
    #     n_params[layer.name] = L.count_params(layer)
    # print(sum(n_params.values()))

    return net, sigma


def build_2dconv_clf_small(config, **kwargs):
    """
    """
    # activation setting
    exec(
        "from lasagne.nonlinearities import {}".format(
        config.hyper_parameters.activation)
    )
    non_lin = eval(config.hyper_parameters.activation)

    # network dict
    net = OrderedDict()
    net, sigma = input_block(net, config)
    net = conv_block(
        net, n_convs=1, n_filters=16, filter_size=(5,5),
        stride=(2,2), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv1', verbose=True
    )
    net = conv_block(
        net, n_convs=1, n_filters=32, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv2', verbose=True
    )
    net = conv_block(
        net, n_convs=2, n_filters=64, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv3', verbose=True
    )
    net = conv_block(
        net, n_convs=2, n_filters=128, filter_size=(3,3),
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv4', verbose=True
    )
    net = conv_block(
        net, n_convs=3, n_filters=256, filter_size=[(3,3),(3,3),(1,1)],
        stride=(1,1), pool_size=(2,2), nonlinearity=non_lin,
        batch_norm=True, name='conv5', verbose=True
    )
    net = output_block(net, config, non_lin)

    # n_params = {}
    # for layer in L.get_all_layers(net['out']):
    #     n_params[layer.name] = L.count_params(layer)
    # print(sum(n_params.values()))

    return net, sigma
