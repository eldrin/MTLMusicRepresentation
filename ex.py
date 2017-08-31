from train import Trainer
from utils import config as CONFIG

def train():
    """"""
    # initialize trainer
    trainer = Trainer(CONFIG)

    # currently, data servers need to be launched manually
    # also, tensor board is needed to be launched manually
    # TODO: automate data servers / tensorboard launching / closing process
    trainer.fit()

if __name__ == "__main__":
    train()



