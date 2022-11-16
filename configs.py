import json
from abc import ABC


def deep_fashion_id_extraction_fn(image_name):
    return image_name.split('/')[-2]


class Config(ABC):
    """ Abstract class for dataset configs """
    pass


class InFashionConfig:
    name = 'in_fashion'
    by_task_embedding_size = {'id':  2048, 'sub_category': 512}
    by_task_output_size = {'id':  7982, 'sub_category': 17}
    tasks = ['id', 'sub_category']
    tasks_weights = {'id':  1.0, 'sub_category': 1.0}
    # dataset_dir = 'datasets/'
    dataset_dir = '/mnt/nfs/tal/datasets/in_fashion'
    image_based_gt_json = 'datasets/gt_tagging/in_fashion_tags_dict.json'
    dataset_metadata = 'datasets/dataset_metadata_closed_catalog.json'
    backbone_name = 'argus'
    num_classes = 7982
    id_extraction_fn = deep_fashion_id_extraction_fn
    f = open('datasets/seeds/seeds_closed_catalog.json',)
    q_sample_image_names = json.load(f)['seeds']



class InFashionOutshop:
    name = 'in_fashion_outshop'

    by_task_embedding_size = {'id':  2048, 'category': 512}
    by_task_output_size = {'id':  6181, 'category': 40}
    tasks = ['id', 'category']
    tasks_weights = {'id':  1.0, 'category': 1.0}

    # dataset_dir = 'datasets/'
    dataset_dir = '/mnt/nfs/tal/datasets/in_fashion'
    image_based_gt_json = 'datasets/gt_tagging/in_fashion_outshop_tags_dict.json'
    dataset_metadata = 'datasets/dataset_metadata_wild.json'
    backbone_name = 'argus'
    num_classes = 6181
    id_extraction_fn = deep_fashion_id_extraction_fn
    f = open('datasets/seeds/seeds_wild.json',)
    q_sample_image_names = json.load(f)['seeds']


def get_config(name):
    if name == 'in_fashion':
        return InFashionConfig
    elif name == 'in_fashion_outshop':
        return InFashionOutshop()
    else:
        raise ValueError('Invalid dataset name: {}'.format(name))