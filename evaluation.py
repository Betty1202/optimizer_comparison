import logging

import numpy as np
import torch
from datasets import load_metric
from pandas import DataFrame


class Evaluation:
    def __init__(self):
        self.metric = load_metric("accuracy")
        self.opt_time_iter = []
        self.loss_iter = []

        self.idx = [0]  # Start iter idx of each epoch

        self.accuracy_epoch = []
        self.opt_time_epoch = []
        self.loss_epoch = []

    def update_iter(self, batch, outputs, time):
        '''
        Update for one iter
        '''
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

        self.opt_time_iter.append(time)
        self.loss_iter.append(float(outputs.loss))

    def update_epoch(self):
        '''
        Update for one epoch
        '''
        self.accuracy_epoch.append(self.metric.compute()["accuracy"])  # After compute(), metric will init itself.
        self.loss_epoch.append(np.mean(self.loss_iter[self.idx[-1]:]))
        self.opt_time_epoch.append(np.mean(self.opt_time_iter[self.idx[-1]:]))
        self.idx.append(len(self.loss_iter))

        logger = logging.getLogger("Evaluation")
        logger.info(
            f"Acc: {self.accuracy_epoch[-1]}, loss: {self.loss_epoch[-1]}, optimize time: {self.opt_time_epoch[-1]}")

    def save(self, file_name: str):
        '''
        Save evaluation result
        '''
        # Save iter evaluation
        iter_data = {
            "loss": self.loss_iter,
            "opt_time": self.opt_time_iter
        }
        df = DataFrame(iter_data)
        df.to_excel(f"{file_name}_iter.xlsx", index=False)

        # Save epoch evaluation
        epoch_data = {
            "loss": self.loss_epoch,
            "opt_time": self.opt_time_epoch,
            "acc": self.accuracy_epoch
        }
        df = DataFrame(epoch_data)
        df.to_excel(f"{file_name}_epoch.xlsx", index=False)
