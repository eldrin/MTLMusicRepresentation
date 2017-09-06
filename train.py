from collections import Iterable
from itertools import chain
import traceback

import numpy as np

from model.helper import load_check_point, save_check_point
from model.helper import get_train_funcs

from utils.misc import get_loggers, get_in_shape


class Trainer:
    """"""
    def __init__(self, config, data_streams, *args, **kwargs):
        """"""
        self.config = config
        self.logger, self.tblogger = get_loggers(config)

        # assign data streams
        self.data_streams = data_streams
        self.epoch_iterators = self._init_epoch_iterators()

        # load model if exists
        self.logger.info('Loading Model..')

        # load model builder
        exec('from model.architecture import build_{} as network_arch'.format(
            config.hyper_parameters.architecture)
        )

        # load model
        self.net = network_arch(config)
        self.k, self.net = load_check_point(
            self.net, config
        )
        # load trained model
        functions = get_train_funcs(self.net, config)

        # assign instance method
        self.partial_fit = {}
        self.cost = {}
        self.predict = {}
        self.dset_size = {}
        for target in config.target:
            self.partial_fit[target] = functions[target]['train']
            self.cost[target] = functions[target]['valid']
            self.predict[target] = functions[target]['predict']

            self.dset_size[target] = {}
            self.dset_size[target]['train'] = eval(
                'self.config.paths.meta_data.size.{}.train'.format(target))
            self.dset_size[target]['valid'] = eval(
                'self.config.paths.meta_data.size.{}.valid'.format(target))

        # get n_iteration
        self.n_iter = sum(
            list(chain.from_iterable(
                [d.values() for d in self.dset_size.values()]
            ))
        )

    def fit(self):
        """"""
        # main training loop
        self.logger.info('Start Training...')

        n_epoch = self.config.train.n_epoch
        check_point_intv = self.config.train.check_point_intv
        verbose_frq = self.config.train.verbose_frq

        try:
            for n in xrange(n_epoch):
                self.logger.debug('Start {:d}th Epoch'.format(n))
                for _ in xrange(self.n_iter):

                    # save check point
                    if self.k % check_point_intv == 0 and self.k > 0:
                        save_check_point(self.k, self.net, self.config)

                    # SELECT TASK
                    task =  np.random.choice(self.config.target)
                    X, Y, _ = self._get_data(task, 'train')

                    # TODO: later, log it
                    if not self._check_data(X):
                        continue

                    # update
                    self.partial_fit[task](X,Y)

                    if self.k % verbose_frq == 0:
                        loss_tr = self.cost[task](X,Y).item()

                        # get valid samples
                        Xv, Yv, _ = self._get_data(task, 'valid')

                        loss_vl = self.cost[task](Xv,Yv).item()

                        # print to logger
                        self.logger.debug(
                            '{:d}th - {} - loss_tr:{:.5f} loss_vl:{:.5f}'.format(
                                self.k, task, loss_tr, loss_vl
                            )
                        )
                        self.tblogger.log_value('loss_{}_tr'.format(task),loss_tr,self.k)
                        self.tblogger.log_value('loss_{}_vl'.format(task),loss_vl,self.k)

                    self.k += 1

        except KeyboardInterrupt as ke:
            self.logger.error(ke)

        except Exception as e:
            traceback.print_exc()

        finally:
            save_check_point(self.k, self.net, self.config)


    @staticmethod
    def _check_data(X):
        """"""
        state = True
        try:
            if X.ndim<3:
                state = False
            elif isinstance(X,Iterable) and (len(X) == 1):
                state = False
        except Exception as e:
            print(X)
            print(e)
            state = False

        return state


    def _get_epoch_iterator(self, target, dset):
        """"""
        return self.data_streams[target][dset].get_epoch_iterator()


    def _init_epoch_iterators(self):
        it = {}
        for target, dset_streams in self.data_streams.iteritems():
            it[target] = {}
            for dset, stream in dset_streams.iteritems():
                it[target][dset] = stream.get_epoch_iterator()
        return it


    def _get_data(self, target, dset):
        try:
            X, Y, req = next(self.epoch_iterators[target][dset])
        except StopIteration as se:
            self.epoch_iterators[target][dset] = self._get_epoch_iterator(target, dset)
            X, Y, req = next(self.epoch_iterators[target][dset])
        return X, Y, req


def test_trainer(config):
    """"""
    trainer = Trainer(config)

