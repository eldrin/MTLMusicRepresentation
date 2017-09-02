import traceback

from utils import load_check_point, save_check_point
from utils import open_datastream, get_loggers
from utils import get_in_shape

from model import get_clf_model

class Trainer:
    """"""
    def __init__(self,config,*args,**kwargs):
        """"""
        self.config = config
        self.logger, self.tblogger = get_loggers(config)

        # open connection to data server
        self.logger.info('Openning data streams...')
        self.data_train = open_datastream(config, is_train=True)
        self.data_valid = open_datastream(config, is_train=False)

        # load model if exists
        self.logger.info('Loading Model..')

        # load model builder
        exec('from model import build_{} as network_arch'.format(
            config.hyper_parameters.architecture)
        )
        # load model
        self.net = network_arch(config)
        self.k, self.net['out'] = load_check_point(
            self.net['out'], config
        )

        # load trained model
        functions = get_clf_model(self.net, config)[0]

        # assign instance method
        self.partial_fit = functions['train']
        self.cost = functions['valid']
        self.predict = functions['predict']

    def fit(self):
        """"""
        # main training loop
        self.logger.info('Start Training...')

        n_iters = self.config.train.n_epoch
        check_point_intv = self.config.train.check_point_intv
        verbose_frq = self.config.train.verbose_frq

        try:
            for n in xrange(n_iters):
                self.logger.debug('Start {:d}th Iteration'.format(n))
                valid_data_iterator = self.data_valid.get_epoch_iterator()
                for j,data in enumerate(self.data_train.get_epoch_iterator()):

                    # save check point
                    if self.k % check_point_intv == 0 and self.k > 0:
                        save_check_point(self.k, self.net['out'], self.config)

                    X,Y,req = data

                    if X.ndim<3 and int(X)==-1:
                        continue

                    # update
                    self.partial_fit(X,Y)

                    if j % verbose_frq == 0:
                        loss_tr = self.cost(X,Y).item()

                        # get valid samples
                        Xv,Yv,_ = next(valid_data_iterator)
                        loss_vl = self.cost(Xv,Yv).item()

                        # print to logger
                        self.logger.debug(
                            '{:d}th loss_tr:{:.5f} loss_vl:{:.5f}'.format(
                                j,loss_tr,loss_vl
                            )
                        )
                        self.tblogger.log_value('loss_tr',loss_tr,self.k)
                        self.tblogger.log_value('loss_vl',loss_vl,self.k)

                    self.k += 1

        except KeyboardInterrupt as ke:
            self.logger.error(ke)

        except Exception as e:
            # self.logger.error(e)
            traceback.print_exc()

        finally:
            save_check_point(self.k, self.net['out'], self.config)


def test_trainer(config):
    """"""
    trainer = Trainer(config)

