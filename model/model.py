import os
from contextlib import contextmanager
from functools import wraps

import numpy as np

import librosa

from helper import load_check_point, save_check_point
from helper import get_train_funcs
from utils.misc import get_in_shape, load_config, load_test_audio

import fire

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

        # for backward compatibility
        if config.hyper_parameters.architecture == '2dconv_clf_deep':
            architecture = 'Conv2DDeep'
        elif config.hyper_parameters.architecture == '2dconv_clf_small':
            architecture = 'Conv2DSmall'
        else:
            architecture = config.hyper_parameters.architecture

        exec('from architecture import {} as Architecture'.format(architecture))

        # load model
        arch = Architecture(config)
        self.net, self.net_var = arch.build()
        self.corruption_var = self.net_var['sigma']
        self.input_var = self.net_var['inputs']

        # load trained model
        self.iter, self.net = load_check_point(self.net, config)

        # get train ops
        funcs = get_train_funcs(self.net, config, feature_layer=feature_layer,
                                input_vars=self.input_var)

        # assign instance method
        self._partial_fit = {}
        self._cost = {}
        self._predict = {}
        self._feature = {}
        for target in config.target:
            self._partial_fit[target] = funcs[target]['train']
            self._cost[target] = funcs[target]['valid']
            self._predict[target] = funcs[target]['predict']
            self._feature[target] = funcs[target]['feature']

    @check_task
    @check_autoencoder
    def partial_fit(self, task, X, y):
        """"""
        if task == 'self':
            self._partial_fit[task](X[:,0], X[:,1], y)
        else:
            self._partial_fit[task](X, y)
        self.iter += 1

    @check_task
    @check_autoencoder
    def cost(self, task, X, y):
        """"""
        if task == 'self':
            c = self._cost[task](X[:,0], X[:,1], y)
        else:
            c = self._cost[task](X, y)
        return c.item()

    @check_task
    @check_autoencoder
    def predict(self, task, X):
        """"""
        if task == 'self':
            p = self._predict[task](X[:,0], X[:,1])
        else:
            p = self._predict[task](X)
        return p

    @check_task
    def feature(self, X):
        """"""
        return self._feature[task](X)

    def save(self):
        """"""
        save_check_point(self.iter, self.net, self.config)

def test_model(config_fn, out_dir):
    """"""
    config = load_config(config_fn)
    tasks = config.target
    model = Model(config)

    y, sr = load_test_audio(config)
    x = np.repeat(y[None,None,55040*2:55040*3], 2, axis=1)
    X = np.array(
        [np.abs(librosa.stft(y_, n_fft=1024, hop_length=256))
         for y_ in x[0]])
    np.save(os.path.join(out_dir, 'test_input.npy'), X)

    Z = {}
    for task in tasks:
        Z[task] = model.predict(task, x)
        np.save(
            os.path.join(
                out_dir, 'test_recon_{}.npy'.format(task)),
            Z[task]
        )

if __name__ == "__main__":
    # test_model_building()
    fire.Fire(test_model)
