from contextlib import contextmanager
from functools import wraps

from helper import load_check_point, save_check_point
from helper import get_train_funcs
from utils.misc import get_in_shape

def check_task(f):
    @wraps(f)
    def _check_task(self, task, *args, **kwargs):
        """"""
        if task not in self.config.target:
            raise ValueError(
                '[ERROR] {} is not included task!'.format(task)
            )
        return f(self, task, *args, **kwargs)

    return _check_task

def check_autoencoder(f):
    @wraps(f)
    def _check_autoencoder(self, task, *args, **kwargs):
        """"""
        if task == 'self':
            self.corruption_var.set_value(self.corruption_rate)
        res = f(self, task, *args, **kwargs)
        if task == 'self':
            self.corruption_var.set_value(0.)
        return res
    return _check_autoencoder


class Model:
    """"""
    def __init__(
        self, config, feature_layer=None, *args, **kwargs):
        """"""
        self.config = config
        if hasattr(config.hyper_parameters, 'input_noise_scale'):
            self.corruption_rate = config.hyper_parameters.input_noise_scale
        else:
            self.corruption_rate = 0.

        # load model builder
        exec('from architecture import build_{} as network_arch'.format(
            config.hyper_parameters.architecture)
        )

        # load model
        self.net, self.corruption_var = network_arch(config)

        # load trained model
        self.iter, self.net = load_check_point(self.net, config)

        # get train ops
        funcs = get_train_funcs(self.net, config, feature_layer=feature_layer)

        # assign instance method
        self._feature = funcs['feature']
        self._partial_fit = {}
        self._cost = {}
        self._predict = {}
        for target in config.target:
            self._partial_fit[target] = funcs[target]['train']
            self._cost[target] = funcs[target]['valid']
            self._predict[target] = funcs[target]['predict']

    @check_task
    @check_autoencoder
    def partial_fit(self, task, X, y):
        """"""
        self._partial_fit[task](X,y)
        self.iter += 1

    @check_task
    @check_autoencoder
    def cost(self, task, X, y):
        """"""
        return self._cost[task](X, y).item()

    @check_task
    @check_autoencoder
    def predict(self, task, X):
        """"""
        return self._predict[task](X)

    def feature(self, X):
        """"""
        return self._feature(X)

    def save(self):
        """"""
        save_check_point(self.iter, self.net, self.config)
