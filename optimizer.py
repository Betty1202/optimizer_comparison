from torch.optim import *

OPTIMIER = {
    "SGD": SGD,
    "Momentum": SGD,
    "Nesterov": SGD,

    "Adagrad": Adagrad,
    "RMSprop": RMSprop,
    "Adadelta": Adadelta,

    "Adam": Adam,
    "AdamW": AdamW,
    "Adamax": Adamax,

}

DEFAULT_OPTIMIER = list(OPTIMIER.keys())
