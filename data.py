import logging

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def get_data(size: int):
    '''
    Load dataset, tokenize them, and construct dataloader
    :param size: size of training dataset
    :return: train_dataloader, eval_dataloader
    '''
    logger = logging.getLogger("Get data")
    logger.info("load GLUE-sst2 dataset")
    raw_datasets = load_dataset("glue", "sst2")

    logger.info("Tokenize dataset")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    logger.info("Split dataset")
    small_train_dataset = tokenized_datasets["train"].select(range(size))
    small_eval_dataset = tokenized_datasets["validation"].select(range(int(size / 10)))

    logger.info("Construct dataloader")
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)
    return train_dataloader, eval_dataloader
