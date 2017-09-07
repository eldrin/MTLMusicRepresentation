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


class Model:
    """"""
    def __init__(
        self, config, feature_layer=None, *args, **kwargs):
        """"""
        self.config = config

        # load model builder
        exec('from architecture import build_{} as network_arch'.format(
            config.hyper_parameters.architecture)
        )

        # load model
        self.net = network_arch(config)
        self.iter, self.net = load_check_point(
            self.net, config
        )
        # load trained model
        funcs = get_train_funcs(self.net, config, feature_layer=feature_layer)

        if debug:
            debug_funcs = get_debug_funcs(
                self.net, feature_layer, cam_layer, self.config)
        else:
            debug_funcs = None

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
    def partial_fit(self, task, X, y):
        """"""
        self._partial_fit[task](X,y)
        self.iter += 1

    @check_task
    def cost(self, task, X, y):
        """"""
        return self._cost[task](X, y).item()

    @check_task
    def predict(self, task, X):
        """"""
        return self._predict[task](X)

    def feature(self, X):
        """"""
        return self._feature(X)

    def save(self):
        """"""
        save_check_point(self.iter, self.net, self.config)
