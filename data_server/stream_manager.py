from collections import Iterable
from fuel.streams import ServerDataStream

class StreamManager:
    """"""
    def __init__(self, ports, config, *args, **kwargs):
        """"""
        self.config = config
        self.host = config.data_server.host
        self.hwm = config.data_server.hwm

        # open streams
        self.data_streams = {}
        for target, dset_ports in ports.iteritems():
            self.data_streams[target] = {}
            for dset, port in dset_ports.iteritems():
                self.data_streams[target][dset] = ServerDataStream(
                    sources=('raw'), produces_examples=True,
                    port=port, host=self.host, hwm=self.hwm
                )

        # initiate epoch iterators
        self.epoch_iterators = self._init_epoch_iterators()

        # assign instance method
        self.dset_size = {}
        for target in config.target:
            self.dset_size[target] = {}
            self.dset_size[target]['train'] = eval(
                'self.config.paths.meta_data.size.{}.train'.format(target))
            self.dset_size[target]['valid'] = eval(
                'self.config.paths.meta_data.size.{}.valid'.format(target))

        # get n_iteration
        self.n_iter = sum([d['train'] for d in self.dset_size.values()])
        self.n_iter = int(self.n_iter / config.hyper_parameters.batch_size)

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


    def get_data(self, target, dset):
        is_data_bad = True
        while is_data_bad:
            try:
                X, Y, req = next(self.epoch_iterators[target][dset])
            except StopIteration as se:
                self.epoch_iterators[target][dset] = self._get_epoch_iterator(target, dset)
                X, Y, req = next(self.epoch_iterators[target][dset])

            # check until data is clean
            if self._check_data(X):
                is_data_bad = False

        return X, Y, req

