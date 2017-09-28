from collections import Iterable
from itertools import chain
import traceback

import numpy as np

from model.model import Model
from utils.misc import get_loggers, get_in_shape

class Trainer:
    """"""
    def __init__(self, config, data_streams, *args, **kwargs):
        """"""
        self.config = config
        self.logger, self.tblogger = get_loggers(config)

        # assign data streams
        self.dstream = data_streams

        # load model if exists
        self.logger.info('Loading Model..')
        self.model = Model(config)

        # set variables
        self.batch_sz = self.config.hyper_parameters.batch_size
        self.n_subbatch = self.config.hyper_parameters.sub_batch_size
        self.n_epoch = self.config.train.n_epoch

        self.n_iter = self.dstream.n_iter
        self.n_iter = int(self.n_iter / self.n_subbatch) + 1
        self.check_point_intv = self.config.train.check_point_intv
        self.verbose_frq = self.config.train.verbose_frq

    def fit(self):
        """"""
        # main training loop
        self.logger.info('Start Training...')
        try:
            for n in xrange(self.n_epoch):
                self.logger.debug('Start {:d}th Epoch'.format(n))
                for _ in xrange(self.n_iter):

                    # save check point
                    if self.model.iter % self.check_point_intv == 0 and \
                       self.model.iter > 0:
                        self.model.save()

                    # SELECT TASK
                    task =  np.random.choice(self.config.target)
                    X, Y, _ = self.dstream.get_data(task, 'train')

                    # update
                    for st in xrange(0, X.shape[0], self.batch_sz):
                        slc = slice(st, st + self.batch_sz)
                        x, y = X[slc], Y[slc]
                        self.model.partial_fit(task, x, y)

                        if self.model.iter % self.verbose_frq == 0:
                            loss_tr = self.model.cost(task, X, Y)

                            # get valid samples
                            Xv, Yv, _ = self.dstream.get_data(task, 'valid')
                            loss_vl = self.model.cost(task, Xv, Yv)

                            # print to logger
                            self.logger.debug(
                                '{:d}th - {} - loss_tr:{:.5f} loss_vl:{:.5f}'.format(
                                    self.model.iter, task, loss_tr, loss_vl
                                )
                            )
                            self.tblogger.log_value(
                                'loss_{}_tr'.format(task),loss_tr,self.model.iter)
                            self.tblogger.log_value(
                                'loss_{}_vl'.format(task),loss_vl,self.model.iter)

        except KeyboardInterrupt as ke:
            self.logger.error(ke)

        except Exception as e:
            traceback.print_exc()

        finally:
            self.model.save()


def test_trainer(config):
    """"""
    trainer = Trainer(config)

