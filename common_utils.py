import torch
import numpy
import os
import logging
import time
import numpy as np
import random
from datetime import timedelta


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(log_dir, dump=True, log_name=None):
    if not(os.path.exists(log_dir)):
        os.makedirs(log_dir)

    if log_name == None:
        log_name = 'train_log.log'
    else:
        log_name = '{}.log'.format(log_name)

    filepath = os.path.join(log_dir, log_name)
    # Create logger
    log_formatter = LogFormatter()

    if dump:
        # create file handler and set level to info
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to info
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if dump:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info('Created main log at ' + str(filepath))
    return logger


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self, end_epoch_reset=True):
        self.end_epoch_reset = end_epoch_reset
        self.steps = 0
        self.total = 0

    def func(self, *values):
        raise NotImplemented

    def update(self, *values):
        val = self.func(*values)
        self.total += val
        self.steps += 1

    def reset(self):
        self.steps = 0
        self.total = 0

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter(RunningAverage):
    def __init__(self):
        super().__init__()

    def func(self, loss_value):
        return loss_value


def count_parameters(module):
  return sum([p.data.nelement() for p in module.parameters()])


def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
