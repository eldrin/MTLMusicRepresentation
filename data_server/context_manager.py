import os
from contextlib import contextmanager
import subprocess

import namedtupled

from utils.misc import load_config
from data_server.stream_manager import StreamManager


DEFAULT_PORT = 5557
# TODO: Deal with the case of launching remote data server

@contextmanager
def data_context(
    config_fn, which_set, verbose=False):
    """"""
    config = load_config(config_fn)

    # check remote
    if hasattr(config.data_server, 'ports'):
        ports = namedtupled.reduce(config.data_server.ports)
        remote_server = True
    else:
        servers, ports = launch_servers(config_fn, which_set, verbose)
        remote_server = False
    streams = StreamManager(ports, config)

    yield  streams

    # tear down data servers only when it's local
    if not remote_server:
        kill_servers(servers)


def launch_servers(
    config_fn, which_set=['train','valid'], verbose=False):
    """"""
    global DEFAULT_PORT

    config = load_config(config_fn)
    data_servers = {}
    ports = {}
    port = DEFAULT_PORT

    for target in config.target:
        data_servers[target] = {}
        ports[target] = {}

        for dset in which_set:
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

            if verbose:
                data_servers[target][dset] = subprocess.Popen(args)
            else:
                with open(os.devnull, 'w') as devnull:
                    data_servers[target][dset] = subprocess.Popen(
                        args, stdout=devnull, stderr=devnull)
        port += 10

    return data_servers, ports


def kill_servers(servers):
    """"""
    for target, dset_servers in servers.iteritems():
        for dset, server_p in dset_servers.iteritems():
            server_p.kill()
    # TODO: double-check remaining server process & kill them
