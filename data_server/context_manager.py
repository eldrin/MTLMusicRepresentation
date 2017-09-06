import os
from contextlib import contextmanager
import subprocess
import tempfile
import cPickle as pkl

from fuel.streams import ServerDataStream

from utils.misc import load_config


DEFAULT_PORT = 5557
# TODO: Deal with the case of launching remote data server

@contextmanager
def data_context(config_fn, ports=None):
    """"""

    # setup data servers and open pipeline to them
    if ports is None:
        remote_server = False
        servers, ports, config = launch_servers(config_fn)
    else:
        remote_server = True
        config = load_config(config_fn)
    streams = open_datastream(ports, config)

    yield  (config, streams)

    # tear down data servers only when it's local
    if not remote_server:
        kill_servers(servers)


def launch_servers(config_fn):
    """"""
    global DEFAULT_PORT

    data_servers = {}
    ports = {}
    port = DEFAULT_PORT
    config = load_config(config_fn)

    for target in config.target:
        data_servers[target] = {}
        ports[target] = {}

        for dset in ['train', 'valid']:
            if dset=='valid':
                ports[target][dset] = port + 1
            else:
                ports[target][dset] = port

            args = [
                'python', '-m', 'data_server.server',
                '--target', target, '--which-set', dset,
                '--port', str(ports[target][dset]),
                '--config-fn', config_fn
            ]
            # data_servers[target][dset] = subprocess.Popen(args)
            with open(os.devnull, 'w') as devnull:
                data_servers[target][dset] = subprocess.Popen(
                    args, stdout=devnull, stderr=devnull)
        port += 10

    return data_servers, ports, config


def kill_servers(servers):
    """"""
    for target, dset_servers in servers.iteritems():
        for dset, server_p in dset_servers.iteritems():
            server_p.kill()
    # TODO: double-check remaining server process & kill them


def open_datastream(ports, config):
    """"""
    host = config.data_server.host
    hwm = config.data_server.hwm

    if host != 'localhost':
        raise ValueError(
            '[ERROR] remote server is not supported yet!'
        )

    data_streams = {}
    for target, dset_ports in ports.iteritems():
        data_streams[target] = {}
        for dset, port in dset_ports.iteritems():
            data_streams[target][dset] = ServerDataStream(
                sources=('raw'), produces_examples=True,
                port=port, host=host, hwm=hwm
            )

    return data_streams

