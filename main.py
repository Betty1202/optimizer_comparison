import argparse
import logging

import torch
from transformers import BertForSequenceClassification

from data import get_data
from optimizer import DEFAULT_OPTIMIER
from trainer import Trainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--optimizer", choices=DEFAULT_OPTIMIER, default='Adam', type=str,
                        help=f"Choose optimizer from {DEFAULT_OPTIMIER}")
    parser.add_argument("--epoch", default=10, type=int,
                        help=f"Epoch number")
    parser.add_argument("--dataset_size", default=1000, type=int,
                        help=f"Size of training dataset")

    return parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger("Main")

    args = get_arguments()
    logger.info(args)

    logger.info("Begin load data")
    train_dataloader, eval_dataloader = get_data(args.dataset_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Begin construct model")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    logger.info("Begin train")
    trainer = Trainer(model, device, args)
    trainer.train(train_dataloader)
    trainer.test(eval_dataloader)
