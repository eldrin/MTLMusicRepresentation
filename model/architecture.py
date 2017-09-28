from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from lasagne import layers as L
from building_block import input_block, conv_block, output_block

class BaseArchitecture(object):
    """"""
    def __init__(self, config, **kwargs):
        """"""
        self._set_conf(config)
        self.net = OrderedDict()
        self.variables = OrderedDict()

    def _set_conf(self, config):
        """"""
        self.config = config

        # activation setting
        exec(
            "from lasagne.nonlinearities import {}".format(
            self.config.hyper_parameters.activation)
        )
        self.non_lin = eval(self.config.hyper_parameters.activation)

        # feature activation setting
        if hasattr(self.config.hyper_parameters, 'feature_activation'):
            exec(
                "from lasagne.nonlinearities import {}".format(
                self.config.hyper_parameters.feature_activation)
            )
            self.feat_non_lin = eval(
                self.config.hyper_parameters.feature_activation)
        else:
            self.feat_non_lin = self.non_lin

        if hasattr(self.config.hyper_parameters, 'size_multiplier'):
            self.m = self.config.hyper_parameters.size_multiplier
        else:
            self.m = 1

    @abstractmethod
    def build(self):
        """"""
        pass

class ConvArchitecture(BaseArchitecture):
    """"""
    def __init__(self, config, **kwargs):
        """"""
        super(ConvArchitecture, self).__init__(config, **kwargs)

        # set architecture configuration
        self.n_filters = []
        self.filter_sizes = []
        self.strides = []
        self.pool_sizes = []
        self.batch_norm = False
        self.verbose = False

    def build(self):
        """"""
        # apply size multiplier
        self.n_filters = map(lambda x:x * self.m, self.n_filters)

        # network dict & set input block
        self.net, self.variables['sigma'] = input_block(self.net, self.config)

        # set conv blocks
        for i, (n_conv, n_filter, flt_sz, stride, pool_sz) in enumerate(zip(
            self.n_convs, self.n_filters, self.filter_sizes, self.strides, self.pool_sizes)):

            if i == (len(self.n_filters)-1):
                non_lins = [self.non_lin] * (n_conv-1)  + [self.feat_non_lin]

            self.net = conv_block(
                self.net, n_convs=n_conv, n_filters=n_filter,
                filter_size=flt_sz, stride=stride, pool_size=pool_sz,
                nonlinearity=self.non_lin, batch_norm=self.batch_norm,
                name='conv{:d}'.format(i), verbose=self.verbose)

        # set output block
        self.net, self.variables['inputs'] = \
                output_block(self.net, self.config, self.non_lin)

        return self.net, self.variables


class Conv2DDeep(ConvArchitecture):
    """"""
    def __init__(self, config, **kwargs):
        """"""
        super(Conv2DDeep, self).__init__(config, **kwargs)

        # set architecture configuration
        self.n_convs = [1, 1, 2, 3, 3, 3]
        self.n_filters = [64, 128, 256, 384, 512, 512]
        self.filter_sizes = [(5,5), (3,3), (3,3), (3,3), (3,3),
                             [(3,3), (3,3), (1,1)]]
        self.strides = [(2,2), None, None, None, None, None]
        self.pool_sizes = [(2,2), (2,2), (2,2), (2,2), (2,2), None]
        self.batch_norm = True
        self.verbose = True


class Conv2DSmall(ConvArchitecture):
    """"""
    def __init__(self, config, **kwargs):
        """"""
        super(Conv2DSmall, self).__init__(config, **kwargs)

        # set architecture configuration
        self.n_convs = [1, 1, 2, 2, 3]
        self.n_filters = [16, 32, 64, 128, 256]
        self.filter_sizes = [(5,5), (3,3), (3,3), (3,3),
                             [(3,3), (3,3), (1,1)]]
        self.strides = [(2,2), None, None, None, None]
        # TODO: last pooling should be None, since GAP follows right after
        self.pool_sizes = [(2,2), (2,2), (2,2), (2,2), (2,2)]
        self.batch_norm = True
        self.verbose = True
