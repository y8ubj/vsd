from collections import OrderedDict
import json
from PIL import Image, ImageFile, ImageFilter
from functools import partial

import torch
import numpy as np
import random
import logging
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGES_FIELD_NAME     = "images"
PATH_FIELD_NAME       = "path"
BBOX_FIELD_NAME       = "bbox"
ATTRIBUTES_FIELD_NAME = 'attributes'
ID_FIELD_NAME         = 'id'
PHASE_FIELD_NAME      = 'phase'
NUM_OF_ATTRIBUTES     = 1000


def dummy_transform(x):
    return x


def double_dummy_transform(x, y):
    return x


class MultitaskConfig:
    """
    Wraps metadata for multitask training configuration. Each task configuration can be registered here and then values of fields can be retrieved for
    all of the registered tasks.
    """

    def __init__(self):
        self.tasks_config = {}

    def add_task_metadata(self, task_name, config):
        """
        Adds the task configuration.
        :param task_name: task name.
        :param config: dictionary with the configuration for the task.
        """
        self.tasks_config[task_name] = config

    def get_by_task_values(self, field_name):
        """
        Gets a dictionary of by task values, where the keys are task names and the values are the values of the given field name for each task.
        :param field_name: name of a field in the task's metadata.
        :return: dictionary of by task field values.
        """
        return {task_name: config[field_name] for task_name, config in self.tasks_config.items()}


class ImageDataset(Dataset):
    """
    Image dataset that is based on a JSON metadata file. The metadata file is an array where each entry includes the details
    of an image in the dataset. The image details should include the relative path to the image from the dataset directory
    and any other wanted fields.
    """
    def __init__(self, root_dir, metadata_path, num_classes=NUM_OF_ATTRIBUTES,
                 crop_bbox=True, im_size=224, with_metadata_transform=None, transform=None, metadata_transform=None):

        self.root_dir = root_dir
        self.metadata_path = metadata_path
        self.im_size = im_size
        self.num_classes = num_classes
        self.crop_bbox = crop_bbox

        self.with_metadata_transform = with_metadata_transform if with_metadata_transform else double_dummy_transform
        self.transform = transform if transform else dummy_transform
        self.metadata_transform = metadata_transform if metadata_transform else dummy_transform

        with open(self.metadata_path) as f:
            self.images_metadata = json.load(f)[IMAGES_FIELD_NAME]

    def __preprocess_image(self, image, image_metadata):
        if len(image.mode) == 1:
            image = Image.fromarray(np.dstack((image, image, image)))

        if len(image.mode) > 3:
            image = Image.fromarray(np.array(image)[:, :, :3])

        if self.crop_bbox:
            bbox = image_metadata[BBOX_FIELD_NAME]
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            return image.crop((x1, y1, x2, y2))
        else:
            return image

    def __preprocess_image_transform(self, image, metadata):
        image = self.with_metadata_transform(image, metadata)
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        image_metadata = self.images_metadata[index]
        path = image_metadata[PATH_FIELD_NAME]
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        print
        image = self.__preprocess_image(image, image_metadata)

        return path, self.__preprocess_image_transform(image, image_metadata), self.metadata_transform(image_metadata)

    def __len__(self):
        return len(self.images_metadata)


class InMemoryDataset(torch.utils.data.Dataset):
    """
    An in memory wrapper for an existing dataset. Loads all of its items into memory for quick access.
    """

    def __init__(self, dataset, transform=dummy_transform):
        super().__init__()
        self.wrapped_dataset = dataset
        self.__dict__.update(dataset.__dict__)
        self.in_memory_data = self.__load_dataset()
        self.transform = transform

    def __load_dataset(self):
        data = [None] * len(self.wrapped_dataset)
        for i in range(len(self.wrapped_dataset)):
            data[i] = self.wrapped_dataset[i]
        return data

    def __getitem__(self, index):
        return self.transform(self.in_memory_data[index])

    def __len__(self):
        return len(self.in_memory_data)


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


class SubsetImageDataset(torch.utils.data.Dataset):
    """
    Subset wrapper for ImageDataset. Uses only subset of the dataset with the given indices.
    """

    def __init__(self, image_dataset, indices, transform=None):
        self.image_dataset = image_dataset
        self.original_indices = indices
        self.transform = transform if transform else dummy_transform
        self.images_metadata = [self.image_dataset.images_metadata[i] for i in indices]
        self.labels = None

    def __getitem__(self, index):
        if self.labels is not None:
            out = self.transform(self.image_dataset[self.original_indices[index]])
            return out[0], out[1], self.labels[index]
        return self.transform(self.image_dataset[self.original_indices[index]])

    def __len__(self):
        return len(self.images_metadata)


class FilteredImageDataset(SubsetImageDataset):
    """
    Filter wrapper for ImageDataset. Filters such the dataset contains only matching images (match is done according to their metadata).
    """

    def __init__(self, image_dataset, filter, transform=None):
        self.image_dataset = image_dataset
        self.filter = filter
        relevant_indices = self.__create_relevant_images_indices()
        super().__init__(image_dataset, relevant_indices, transform if transform else dummy_transform)

    def __create_relevant_images_indices(self):
        relevant_images_indices = []
        for index, image_metadata in enumerate(self.image_dataset.images_metadata):
            if self.filter(image_metadata):
                relevant_images_indices.append(index)

        return relevant_images_indices


class PhaseFilteredImageDataset(SubsetImageDataset):
    """
    Filter wrapper for ImageDataset. Filters such the dataset contains only matching images (match is done according to their metadata).
    """

    def __init__(self, image_dataset, phases_list, transform=dummy_transform):
        self.image_dataset = image_dataset
        self.phases_list = phases_list
        self.transform = transform
        relevant_indices = self.__create_relevant_images_indices()
        super().__init__(image_dataset, relevant_indices)

    def __create_relevant_images_indices(self):
        relevant_images_indices = []
        for index, image_metadata in enumerate(self.image_dataset.images_metadata):
            if image_metadata['phase'] in self.phases_list:
                relevant_images_indices.append(index)

        return relevant_images_indices

    def __getitem__(self, index):
        im_name, im, y = self.image_dataset[self.original_indices[index]]
        out_image = self.transform(im)
        return im_name, out_image, y


def create_query_gallery_dataset_split(image_dataset, num_in_query=1, num_in_gallery=100, split_field="id", rnd_state=None):
    # num_in_gallery=100 by defulte not to limit the items
    id_options_mapper = FieldOptionsMapper(image_dataset.images_metadata, split_field)
    query_indices = []
    gallery_indices = []

    for id_indices in id_options_mapper.by_field_indices():
        replace = num_in_query > len(id_indices)
        if rnd_state:
            for_query_indices = rnd_state.choice(id_indices, num_in_query, replace=replace)
        else:
            for_query_indices = np.random.choice(id_indices, num_in_query, replace=replace)
        query_indices.extend(set(for_query_indices.tolist()))
        gallery_indices.extend([i for i in id_indices if i not in for_query_indices][:num_in_gallery])
    return SubsetImageDataset(image_dataset, query_indices), SubsetImageDataset(image_dataset, gallery_indices)


def filter_func(field_name, field_values, image_metadata):
    return image_metadata[field_name] in field_values


def create_field_filtered_image_dataset(image_dataset, field_name, field_values):
    """
    Creates an image dataset that is a subset of the given dataset with only the images that have a matching value for the given field name and
    values.
    :param image_dataset: image dataset.
    :param field_name: name of the field to filter by.
    :param field_values: sequence of matching field values. An item will be in the filtered dataset if its field value is in this sequence.
    :return: FilteredImageDataset with only the images with a field value that is in the given field values for the field name.
    """

    return FilteredImageDataset(image_dataset, partial(filter_func, field_name, field_values))


def create_label_mapper(image_dataset, field_name):
    """
    Creates a dictionary mapping between a label to its index.
    :param image_dataset: image dataset.
    :param field_name: field name to create the mapper for.
    :return: dictionary of field value (label) to id.
    """
    return create_label_mapper_for_metadatas(image_dataset.images_metadata, field_name)


def create_label_mapper_for_metadatas(metadatas, field_name):
    """
    Creates a dictionary mapping between a label to its index.
    :param metadatas: sequence of image metadatas.
    :param field_name: field name to create the mapper for.
    :return: dictionary of field value (label) to id.
    """
    labels_set = {metadata[field_name] for metadata in metadatas if field_name in metadata}
    sorted_labels = sorted(list(labels_set))
    return {label: i for i, label in enumerate(sorted_labels)}


def create_frequent_values_filtered_image_dataset(image_dataset, field_name, freq_threshold=2):
    """
    Creates a FilteredImageDataset, filtering out images that have infrequent values for a certain field. The resulting dataset will contain only
    images that their value for the field has frequency that is greater than (or equals) to freq_threshold.
    :param image_dataset: image dataset.
    :param field_name: name of the field to filter according to its frequency.
    :param freq_threshold: threshold frequency to keep images with values that their frequency is above (or equal) to the threshold.
    :return: FilteredImageDataset with images with frequent values for the given field.
    """
    field_indices_mapper = FieldOptionsMapper(image_dataset.images_metadata, field_name)
    indices_to_remove = set()
    for field_value, indices in field_indices_mapper.field_value_to_indices.items():
        if len(indices) < freq_threshold:
            indices_to_remove.update(indices)

    indices_to_keep = [i for i in range(len(image_dataset)) if i not in indices_to_remove]
    return SubsetImageDataset(image_dataset, indices_to_keep)


def create_dict_of_label_mappers(image_dataset, by_task_field_name):
    """
    Creates dictionary of mappers that map field value to label id. Each key in field_names_dict is in the returned dictionary and the value is the
    mapper relevant to that field.
    :param image_dataset: image dataset.
    :param by_task_field_name: list of tasks
    :return: dictionary of task name to label mapper.
    """
    label_mappers = OrderedDict()
    for task_name in by_task_field_name:
        label_mappers[task_name] = create_label_mapper(image_dataset, task_name)

    return label_mappers


def metadata_to_label_ids(label_mappers, by_task_field_name, no_label_id, metadata):
    label_ids_dict = OrderedDict()
    for task_name in by_task_field_name:
        label_mapper = label_mappers[task_name]

        if task_name not in metadata:
            label_ids_dict[task_name] = no_label_id
            continue

        field_value = metadata[task_name]
        if field_value not in label_mapper:
            label_ids_dict[task_name] = no_label_id
            continue

        label_ids_dict[task_name] = label_mapper[field_value]

    return label_ids_dict


def create_multitask_extract_label_ids_metadata_transform(image_dataset, by_task_field_name, no_label_id=-100):
    """
    Creates a metadata transform function that extracts a dictionary of labels for the given ids. The keys of the field_names_dict will be those
    of the returned dictionary from the transform and the values will be the label ids.
    :param image_dataset: image dataset.
    :param by_task_field_name: list of tasks
    :param no_label_id: default id the transform will return to images without the given field name.
    :return: metadata transform to extract multitask label ids.
    """

    label_mappers = create_dict_of_label_mappers(image_dataset, by_task_field_name)
    return partial(metadata_to_label_ids, label_mappers, by_task_field_name, no_label_id)


def transforms_regular(im_size=224):
    return transforms.Compose([transforms.Resize((im_size, im_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def transforms_augmentation_with_bbox(im_size=224):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.Resize((im_size, im_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def _convert_image_to_rgb(image):
    return image.convert("RGB")

moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


def costume_collate_fn(batch):
    x1 = []
    x2 = []
    for elem in batch:
        x1.append(elem[0])
        x1.append(elem[2])
        x2.append(elem[1])
        x2.append(elem[3])
    x1 = torch.stack(x1)
    x2 = torch.stack(x2)
    return [x1, x2]


class MultiTaskDatasetFactory():

    def __init__(self, dataset_metadata,
                 crop_bbox=False,
                 image_size=224,
                 in_memory=False,
                 id_freq_threshold=1,
                 rnd_state=None):

        self.dataset_metadata = dataset_metadata
        self.crop_bbox = crop_bbox
        self.image_size = image_size
        self.in_memory = in_memory
        self.id_freq_threshold = id_freq_threshold
        if rnd_state != None:
            self.rnd_state = np.random.RandomState(rnd_state)
        else:
            self.rnd_state = rnd_state


    def get_train_and_query_gallery_datasets(self,
                                             use_all_data=False,
                                             num_in_gallery=100,
                                             transforms_train=transforms_augmentation_with_bbox,
                                             transforms_test=transforms_regular):

        image_dataset = ImageDataset(root_dir=self.dataset_metadata.dataset_dir,
                                     metadata_path=self.dataset_metadata.dataset_metadata,
                                     num_classes=self.dataset_metadata.num_classes,
                                     crop_bbox=self.crop_bbox,
                                     im_size=self.image_size,
                                     transform=None,
                                     metadata_transform=None)

        if self.in_memory:
            image_dataset = InMemoryDataset(image_dataset)

        if use_all_data:
            train_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["train", "test", "query", "gallery"])
        else:
            train_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["train"])

        train_dataset = create_frequent_values_filtered_image_dataset(train_dataset, "id",
                                                                            self.id_freq_threshold)

        test_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["test", "search"])
        test_dataset_2 = create_field_filtered_image_dataset(image_dataset, "phase", ["train"])
        # test_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["train", "test", "search"])
        query_dataset, gallery_dataset = create_query_gallery_dataset_split(test_dataset,
                                                                            num_in_query=1,
                                                                            num_in_gallery=num_in_gallery,
                                                                            split_field="id",
                                                                            rnd_state=self.rnd_state)

        train_metadata_transform = create_multitask_extract_label_ids_metadata_transform(train_dataset,
                                                                                         self.dataset_metadata.tasks)

        val_metadata_transform = create_multitask_extract_label_ids_metadata_transform(train_dataset,
                                                                                       self.dataset_metadata.tasks)
        train_transform = partial(generate_transform, train_metadata_transform, transforms_train)
        val_transform = partial(generate_transform, val_metadata_transform, transforms_test)


        train_dataset.transform = train_transform
        gallery_dataset.transform = val_transform
        query_dataset.transform = val_transform
        test_dataset.transform = val_transform
        test_dataset_2.transform = val_transform

        # return train_dataset, query_dataset, gallery_dataset
        # return train_dataset, test_dataset, test_dataset_2 # FOR SEARCH
        return train_dataset, train_dataset, train_dataset 

    def get_train_and_query_gallery_datasets_outshop(self,
                                             use_all_data=False,
                                             num_in_gallery=100,
                                             transforms_train=transforms_augmentation_with_bbox,
                                             transforms_test=transforms_regular):

        image_dataset = ImageDataset(root_dir=self.dataset_metadata.dataset_dir,
                                     metadata_path=self.dataset_metadata.dataset_metadata,
                                     num_classes=self.dataset_metadata.num_classes,
                                     crop_bbox=self.crop_bbox,
                                     im_size=self.image_size,
                                     transform=None,
                                     metadata_transform=None)

        if self.in_memory:
            image_dataset = InMemoryDataset(image_dataset)

        if use_all_data:
            train_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["train", "test"])
        else:
            train_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["train"])

        train_dataset = create_frequent_values_filtered_image_dataset(train_dataset, "id",
                                                                            self.id_freq_threshold)

        test_dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["search"])

        query_dataset, gallery_dataset = create_query_gallery_dataset_split(test_dataset,
                                                                            num_in_query=1,
                                                                            num_in_gallery=num_in_gallery,
                                                                            split_field="id",
                                                                            rnd_state=self.rnd_state)

        train_metadata_transform = create_multitask_extract_label_ids_metadata_transform(train_dataset,
                                                                                         self.dataset_metadata.tasks)

        val_metadata_transform = create_multitask_extract_label_ids_metadata_transform(train_dataset,
                                                                                       self.dataset_metadata.tasks)
        train_transform = partial(generate_transform, train_metadata_transform, transforms_train)
        val_transform = partial(generate_transform, val_metadata_transform, transforms_test)


        train_dataset.transform = train_transform
        gallery_dataset.transform = val_transform
        query_dataset.transform = val_transform
        test_dataset.transform = val_transform

        # return train_dataset, query_dataset, gallery_dataset
        # return train_dataset, test_dataset, test_dataset_2 # FOR SEARCH
        return train_dataset, test_dataset, train_dataset 


    def get_train_and_query_gallery_data_loaders(self, use_all_data=False, batch_size_train=128, batch_size_test=128, num_workers=0, shuffle_test=False, num_in_gallery=100, transforms_train=transforms_regular):
        train_dataset, query_dataset, gallery_dataset = self.get_train_and_query_gallery_datasets(use_all_data, num_in_gallery=num_in_gallery, transforms_train=transforms_train)
        logging.info('Created dataset, train has {} images, query has {} images, gallery has {} images'.format(
            len(train_dataset), len(query_dataset), len(gallery_dataset)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True)
        query_loader = DataLoader(query_dataset, batch_size=batch_size_test, num_workers=num_workers, shuffle=shuffle_test)
        gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size_test, num_workers=num_workers, shuffle=shuffle_test)
        return train_loader, query_loader, gallery_loader


    def get_train_and_query_gallery_data_loaders_outshop(self, use_all_data=False, batch_size_train=128, batch_size_test=128, num_workers=0, shuffle_test=False, num_in_gallery=100, transforms_train=transforms_regular):
        train_dataset, query_dataset, gallery_dataset = self.get_train_and_query_gallery_datasets_outshop(use_all_data, num_in_gallery=num_in_gallery, transforms_train=transforms_train)
        logging.info('Created dataset, train has {} images, query has {} images, gallery has {} images'.format(
            len(train_dataset), len(query_dataset), len(gallery_dataset)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True)
        query_loader = DataLoader(query_dataset, batch_size=batch_size_test, num_workers=num_workers, shuffle=shuffle_test)
        gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size_test, num_workers=num_workers, shuffle=shuffle_test)
        return train_loader, query_loader, gallery_loader


# partial
def generate_transform(train_metadata_transform, transforms_augmentation_with_bbox, x_and_metadata):
    return (x_and_metadata[0],
    #  dummy_transform(x_and_metadata[1]),
     transforms_augmentation_with_bbox()(x_and_metadata[1]),
     train_metadata_transform(x_and_metadata[2]))


def get_dataset_factory(dataset_metadata, crop_bbox=False, image_size=224, in_memory=False, id_freq_threshold=1, rnd_state=None):
        return MultiTaskDatasetFactory(dataset_metadata, crop_bbox, image_size, in_memory, id_freq_threshold, rnd_state)