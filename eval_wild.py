""" Simple script for testing a trained VBS model and view some results """
import torch
from datasets.configs import *
from evaluation.metrics import *
from model_vsd import Dino, Argus
from datasets.configs import get_config
import utils_vsd as utils
from datasets.image_dataset import get_dataset_factory
import os

class Params(utils.ParamsBase):

    # Experiment
    seed = 0
    folder_name = 'Dino'
    dump_log = False
    gpu_num = 1

    checkpoint = None

    # Data
    dataset = 'in_fashion_outshop'
    use_all_data = False
    include_query_id = True # False = FingerPrinting, True = GT

    # Eval
    filter_by_metadata_field = False
    filter_unique_ids = True


    metadata_field = 'sub_category'

    log_dir = '{}/{}'.format(dataset, folder_name)
    in_memory = False
    id_freq_threshold = 1
    
    crop_bbox = False


    # Model
    backbone = 'argus'

    batch_size_train = 512
    batch_size_test = 128
    num_workers = 4
    pin_memory = True

    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    params = Params()
    utils.create_logger(os.path.abspath(__file__ + '/../'), dump=False)

    dataset_metadata = get_config(params.dataset)
    dataset_factory = get_dataset_factory(dataset_metadata=dataset_metadata,
                                          crop_bbox=params.crop_bbox,
                                          image_size=224,
                                          in_memory=params.in_memory,
                                          id_freq_threshold=params.id_freq_threshold,
                                          rnd_state=params.seed)


    train_loader, query_loader, gallery_loader = dataset_factory.get_train_and_query_gallery_data_loaders_outshop(use_all_data=params.use_all_data,
                                                                                                          batch_size_train=params.batch_size_train,
                                                                                                          batch_size_test=params.batch_size_train,
                                                                                                          num_workers=params.num_workers,                                                                                        shuffle_test=False)                                                                                                          
    # model = Dino()
    model = Argus()

    model = model.to(params.device)
    model.eval()                        

    gt_image_retrieval_acc = GT_ImageBasedRetrievalAccuracyCalculator(path_to_gt_json=dataset_metadata.image_based_gt_json,
                                                                      ks_hr=[5, 9],
                                                                      ks_mrr=[5, 9],
                                                                      device=params.device,
                                                                      distance_func_name='cosine')


    mean_acc_at_k_gt, mrr_at_k, mrr, roc_auc, pr_auc = gt_image_retrieval_acc.calc(model=model,
                                                                  query_loader=query_loader,
                                                                  gallery_loader=gallery_loader,
                                                                  dataset_metadata=dataset_metadata,
                                                                  filter_by_metadata_field=params.filter_by_metadata_field,
                                                                  metadata_field=params.metadata_field,
                                                                  verbose=True, include_query_id=params.include_query_id)
    for k, v in mean_acc_at_k_gt.items():
        print('HR@{}: {:.2f}'.format(k, v['mean'] * 100))
    for k, v in mrr_at_k.items():
        print('MRR@{}: {:.2f}'.format(k, v['mean'] * 100))
    # print('MRR: {:.2f}'.format(mrr['mean'] * 100))
    print('ROC-AUC: {:.2f}'.format(roc_auc * 100))
    print('PR-AUC: {:.2f}'.format(pr_auc * 100))