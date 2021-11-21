from torch.optim import *

OPTIMIER = {
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "Adam": Adam,
    "AdamW": AdamW,
    "Adamax": Adamax,
    "ASGD": ASGD,
    # "LBFGS": LBFGS,
    "RMSprop": RMSprop,
    "Rprop": Rprop,
    "SGD": SGD
}

DEFAULT_OPTIMIER = list(OPTIMIER.keys())
