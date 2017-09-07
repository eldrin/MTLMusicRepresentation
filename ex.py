import os

from train import Trainer
from data_server.context_manager import data_context

import fire

# TODO: automate tensorboard launching / closing process
#       currently, tensor board is needed to be launched manually

def main(config_fn, data_verbose=False):
    """"""
    # load config file
    config_fn = os.path.abspath(config_fn)
    config = load_config(config_fn)

    # launch data servers
    with data_context(config, data_verbose) as data_streams:

        # initialize trainer
        trainer = Trainer(config, data_streams)

        # train
        trainer.fit()

if __name__ == "__main__":
    fire.Fire(main)
