from timeit import default_timer as timer

import torch
from tqdm.auto import tqdm
from transformers import get_scheduler

from evaluation import Evaluation
from optimizer import OPTIMIER


class Trainer:
    def __init__(self, model, device: torch.device, args):
        self.device = device
        self.model = model.to(device)
        self.optimizer = OPTIMIER[args.optimizer](model.parameters(), 5e-5)
        self.epochs_num = args.epoch

    def train(self, dataloader):
        num_training_steps = self.epochs_num * len(dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        evaluation = Evaluation()
        for epoch in range(self.epochs_num):
            self.iteration(dataloader, progress_bar, lr_scheduler, evaluation, True)
        return evaluation

    def test(self, dataloader):
        num_training_steps = self.epochs_num
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        evaluation = Evaluation()
        self.iteration(dataloader, progress_bar, lr_scheduler, evaluation, False)
        return evaluation

    def iteration(self, dataloader, progress_bar, lr_scheduler, evaluation: Evaluation, is_train: bool):
        self.model.train(is_train)
        with torch.set_grad_enabled(is_train):
            for batch in dataloader:
                # Send data to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Calculate model
                outputs = self.model(**batch)
                if is_train:
                    loss = outputs.loss
                    loss.backward()

                    # Optimize
                    optimizer_start = timer()
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    optimizer_end = timer()

                # Update for progress and evaluation
                progress_bar.update(1)
                evaluation.update_iter(batch, outputs, optimizer_end - optimizer_start if is_train else 0)

                del batch
            evaluation.update_epoch()
        return evaluation
