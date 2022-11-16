import logging
import os
import time
import numpy as np
import torch
import random
import json
import collections
import copy
from datetime import timedelta


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


class Accuracy():

    def __init__(self, end_epoch_reset=True):
        self.end_epoch_reset = end_epoch_reset
        self.correct = 0.0
        self.total = 0.0

    def func(self, *values):
        raise NotImplemented

    def update(self, outputs, labels):
        outputs = outputs.detach().cpu()
        labels = labels.detach().cpu()

        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def reset(self):
        self.correct = 0.0
        self.total = 0.0

    def __call__(self):
        return self.correct / self.total


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


def create_logger(log_dir, dump=True):
    filepath = os.path.join(log_dir, 'net_launcher_log.log')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # Safety check
    # if os.path.exists(filepath) and opt.checkpoint == "":
    #     logging.warning("Experiment already exists!")

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


def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def load_checkpoint(checkpoint, model, params, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    if params.device.type == 'cpu':
        checkpoint = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_tensor(x, file_name):
    torch.save(x, file_name)


def dump_to_json(dict, name_str, dir=None):
    file = json.dumps(dict)
    if dir:
        file_name = os.path.join(dir, name_str + ".json")
    else:
        file_name = name_str + ".json"

    f = open(file_name, "w")
    f.write(file)
    f.close()


class ParamsBase():

    def str(self):
        attrs = [item for item in self.__dir__() if not item.startswith('_')]
        str = 'Params:: ' + ''.join(['{} : {}\n'.format(at, getattr(self, at)) for at in attrs])
        return str

    def todict(self):
        attrs = [item for item in self.__dir__() if not item.startswith('_')]
        d = {at: getattr(self, at) for at in attrs}
        d.pop('str')
        d.pop('todict')
        d.pop('save')
        return d

    def save(self):
        r = [item.split(' : ') for item in self.str().split('\n')]
        d = {i[0]: i[1] for i in r[:-1]}
        with open(self.log_dir + '/params.json', 'w') as f:
            json.dump(d, f)


def flatten_dicts(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dicts(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ModelLoadingValidator(object):
    def __init__(self, org_model, num_layer_to_sample=0):

        self.num_layer_to_sample = num_layer_to_sample
        self.org_w = self.sample_weights(org_model)

    def sample_weights(self, model):
        sample_w = copy.deepcopy(list(model.parameters())[self.num_layer_to_sample])
        if not isinstance(sample_w, torch.nn.parameter.Parameter):
            raise ValueError('Saved item is not from type Parameter, is type : {}'.format(type(sample_w)))
        return sample_w

    def validate(self, updates_model):
        updated_w = self.sample_weights(updates_model)
        try:
            assert self.org_w.shape == updated_w.shape
        except AssertionError as error:
            AssertionError('Original and updated tensor have different shapes')

        diff = updated_w.eq(self.org_w).all()
        # if the two tensors are equal - raise error
        if diff:
            logging.info('Model loading failed ! - recheck code')
            raise AssertionError
        else:
            logging.info('Model loading succeeded')


class FieldOptionsMapper:
    """
    Maps a given index to the indices of all items with the same field value.
    """

    def __init__(self, metadatas, field_name, allow_missing_field=False):
        self.metadatas = metadatas
        self.field_name = field_name
        self.field_value_to_indices = {}
        self.allow_missing_field = allow_missing_field

        for index, metadata in enumerate(self.metadatas):
            field_value = self.__get_metadata_field_value(metadata)
            if field_value not in self.field_value_to_indices:
                self.field_value_to_indices[field_value] = []

            self.field_value_to_indices[field_value].append(index)

    def __get_metadata_field_value(self, metadata):
        if self.field_name in metadata:
            return metadata[self.field_name]
        elif self.allow_missing_field:
            return ""
        else:
            raise ValueError(f"Missing field value for field '{self.field_name}' in metadata: {json.dumps(metadata, indent=2)}")

    def __getitem__(self, index):
        field_value = self.get_field_value(index)
        return self.field_value_to_indices[field_value]

    def by_field_indices(self):
        return self.field_value_to_indices.values()

    def get_field_value(self, index):
        return self.__get_metadata_field_value(self.metadatas[index])

    def get_identifier(self, index):
        return self.get_field_value(index)

    def get_identifier_to_indices_dict(self):
        return self.field_value_to_indices


