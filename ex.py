import os

from train import Trainer
from data_server.context_manager import data_context

import fire

# TODO: automate tensorboard launching / closing process
#       currently, tensor board is needed to be launched manually

def train(config_fn, data_verbose=False):
    """"""
    config_fn = os.path.abspath(config_fn)

    # launch data servers
    with data_context(config_fn, verbose=data_verbose) as (config, data_streams):

        # initialize trainer
        trainer = Trainer(config, data_streams)

        # train
        trainer.fit()

if __name__ == "__main__":
    fire.Fire(train)
