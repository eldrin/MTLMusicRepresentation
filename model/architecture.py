from abc import abstractmethod
from collections import OrderedDict

from lasagne import layers as L
from lasagne.nonlinearities import *
from building_block import input_block, conv_block, output_block
from custom_layer import build_siamese


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
        self.n_filters = map(lambda x: x * self.m, self.n_filters)

        # network dict & set input block
        self.net, self.variables['sigma'] = input_block(self.net, self.config)

        # set conv blocks
        it = zip(self.n_convs, self.n_filters, self.filter_sizes,
                 self.strides, self.pool_sizes)
        for i, (n_conv, n_filter, flt_sz, stride, pool_sz) in enumerate(it):

            # if i == (len(self.n_filters)-1):
            #     non_lins = [self.non_lin] * (n_conv-1) + [self.feat_non_lin]

            self.net = conv_block(
                self.net, n_convs=n_conv, n_filters=n_filter,
                filter_size=flt_sz, stride=stride, pool_size=pool_sz,
                nonlinearity=self.non_lin, batch_norm=self.batch_norm,
                name='conv{:d}'.format(i), verbose=self.verbose)

        # set output block
        self.net, self.variables['inputs'] = output_block(
            self.net, self.config, self.non_lin)

        return self.net, self.variables


class Conv2DDeep(ConvArchitecture):
    """"""
    def __init__(self, config, **kwargs):
        """"""
        super(Conv2DDeep, self).__init__(config, **kwargs)

        # set architecture configuration
        self.n_convs = [1, 1, 2, 3, 3, 3]
        self.n_filters = [64, 128, 256, 384, 512, 512]
        self.filter_sizes = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3),
                             [(3, 3), (3, 3), (1, 1)]]
        self.strides = [(2, 2), None, None, None, None, None]
        self.pool_sizes = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), None]
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
        self.filter_sizes = [(5, 5), (3, 3), (3, 3), (3, 3),
                             [(3, 3), (3, 3), (1, 1)]]
        self.strides = [(2, 2), None, None, None, None]
        self.pool_sizes = [(2, 2), (2, 2), (2, 2), (2, 2), None]
        self.batch_norm = True
        self.verbose = True


class Conv2DSmallChimera(BaseArchitecture):
    """"""
    def __init__(self, config, input_mel=True, **kwargs):
        """"""
        super(Conv2DSmallChimera, self).__init__(config, **kwargs)

        self.mel = input_mel
        self.n_convs = [1, 1, 2, 2, 2, 3]
        self.n_filters = [16, 32, 64, 128, 256, 256]
        self.filter_sizes = [(5, 5), (3, 3), (3, 3), (3, 3),
                             (3, 3), [(3, 3), (3, 3), (1, 1)]]
        self.strides = [(2, 1), None, None, None, None, None]
        self.pool_sizes = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), None]
        self.batch_norm = True
        self.verbose = True

        branch_at = config.hyper_parameters.branch_at
        if isinstance(branch_at, (int, float)):
            if branch_at > len(self.n_convs):
                raise ValueError(
                    '[ERROR] branch point need to be smaller than n_convs')
        elif isinstance(branch_at, str):
            if branch_at != 'fc':
                raise ValueError(
                    '[ERROR] only support "fc" for string branch point now')
        else:
            raise ValueError(
                '[ERROR] branch point must be number or "fc"')
        self.branch_at = branch_at

    def build(self):
        """"""
        # output setting
        out_acts = []
        for out_act in self.config.hyper_parameters.out_act:
            # exec('from lasagne.nonlinearities import {}'.format(out_act))
            out_acts.append(eval(out_act))
        n_outs = self.config.hyper_parameters.n_out
        targets = self.config.target

        # apply size multiplier
        self.n_filters = map(lambda x: x * self.m, self.n_filters)

        # network dict & set input block
        self.net, self.variables['sigma'] = input_block(
            self.net, self.config, melspec=True)

        if self.branch_at == 'fc':
            # shares all layers except output layer
            for i in range(len(self.n_convs)):
                name = 'conv{:d}'.format(i+1)
                self.net = conv_block(
                    self.net, self.n_convs[i], self.n_filters[i],
                    self.filter_sizes[i], self.strides[i],
                    self.pool_sizes[i], self.non_lin,
                    self.batch_norm, self.net.keys()[-1], name)
            # GAP
            self.net['gap'] = L.batch_norm(L.GlobalPoolLayer(
                self.net[next(reversed(self.net))], name='gap'))
            self.net['fc'] = L.dropout(
                L.batch_norm(
                    L.DenseLayer(
                        self.net['gap'],
                        num_units=self.net['gap'].output_shape[-1],
                        nonlinearity=self.non_lin,
                        name='fc'
                    )
                ),
                name='fc.bn.do'
            )

            # Out block
            out_layer_names = []
            for target, n_out, out_act in zip(targets, n_outs, out_acts):
                out_layer_names.append('{}.out'.format(target))
                if target == 'self':
                    self.net[out_layer_names[-1]], inputs = \
                            build_siamese(self.net['fc'])
                else:
                    self.net[out_layer_names[-1]] = L.DenseLayer(
                        self.net['fc'],
                        num_units=n_out,
                        nonlinearity=out_act,
                        name=out_layer_names[-1]
                    )
                    inputs = [self.net['input'].input_var]
        else:
            # shares lower part of the network and branch out

            # shared conv blocks
            for i in range(self.branch_at-1):
                name = 'conv{:d}'.format(i+1)
                self.net = conv_block(
                    self.net, self.n_convs[i], self.n_filters[i],
                    self.filter_sizes[i], self.strides[i],
                    self.pool_sizes[i], self.non_lin,
                    self.batch_norm, name, self.net.keys()[-1],
                    self.verbose)
            branch_point = self.net.keys()[-1]

            # branch out to each targets
            out_layer_names = []
            for target, n_out, out_act in zip(targets, n_outs, out_acts):
                # first conv_block for each branch
                j = self.branch_at-1  # branch_point_ix
                name = '{}.conv{:d}'.format(target, j+1)
                self.net = conv_block(
                    self.net, self.n_convs[j], self.n_filters[j],
                    self.filter_sizes[j], self.strides[j],
                    self.pool_sizes[j], self.non_lin,
                    self.batch_norm, name, branch_point, self.verbose)

                for i in range(self.branch_at, len(self.n_convs)):
                    name = '{}.conv{:d}'.format(target, i+1)
                    self.net = conv_block(
                        self.net, self.n_convs[i], self.n_filters[i],
                        self.filter_sizes[i], self.strides[i],
                        self.pool_sizes[i], self.non_lin,
                        self.batch_norm, name, self.net.keys()[-1],
                        self.verbose)

                # GAP
                gap_name = '{}.gap'.format(target)
                self.net[gap_name] = L.batch_norm(L.GlobalPoolLayer(
                    self.net[next(reversed(self.net))], name=gap_name))

                # FC
                fc_name = '{}.fc'.format(target)
                self.net[fc_name] = L.dropout(
                    L.batch_norm(
                        L.DenseLayer(
                            self.net[gap_name],
                            num_units=self.net[gap_name].output_shape[-1],
                            nonlinearity=self.non_lin,
                            name=fc_name
                        )
                    ),
                    name=fc_name
                )

                # OUT
                out_layer_names.append('{}.out'.format(target))
                if target == 'self':
                    self.net[out_layer_names[-1]], inputs = \
                            build_siamese(self.net[fc_name])
                else:
                    self.net[out_layer_names[-1]] = L.DenseLayer(
                        self.net[fc_name],
                        num_units=n_out,
                        nonlinearity=out_act,
                        name=out_layer_names[-1]
                    )
                    inputs = [self.net['input'].input_var]

        # make a concatation layer just for save/load purpose
        self.net['IO'] = L.ConcatLayer(
            [L.FlattenLayer(self.net[target_layer_name])
             if target == 'self' else self.net[target_layer_name]
             for target_layer_name in out_layer_names],
            name='IO'
        )
        self.variables['inputs'] = inputs

        if self.verbose:
            for target in targets:
                print(self.net['{}.out'.format(target)].output_shape)

        return self.net, self.variables
