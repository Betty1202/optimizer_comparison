from datasets import load_dataset, load_metric
import torch


class Evaluation:
    def __init__(self):
        self.metric = load_metric("accuracy")
        self.opt_time = []
        self.loss = []
        self.accuracy = []

    def update_iter(self, batch, outputs, time):
        '''
        Update for one iter
        '''
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

        self.opt_time.append(time)
        self.loss.append(outputs.loss)

    def update_epoch(self):
        '''
        Update for one epoch
        '''
        self.accuracy.append(self.metric.compute()["accuracy"])
        self.metric = load_metric("accuracy")  # Init metric
