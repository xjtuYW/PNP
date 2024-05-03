import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits


from util.general_utils import AverageMeter, init_experiment, load_trained_paras, set_seed, np_cosine_sim
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import *

from models import vision_transformer as vits

from sklearn.cluster import KMeans
# from finch import FINCH
from cluster import *
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import minimize_scalar
from functools import partial
from scipy.optimize import linear_sum_assignment as linear_assignment


estimated_k_gb_gcd_and_ours = {
    'cifar10': 9,
    'cifar100': 100,
    'imagenet_100':109,
    'cub':231,
    'scars':230,
    'aircraft':102,
    'pets':38,
    'herbarium_19':520
}


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

@torch.no_grad()
def test_kmeans_for_scipy(K, merge_test_loader, model, args=None, verbose=False):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    model.eval()
    K = int(K)

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):
        images = images.to(device)
        feats  = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    print(f'Fitting K-Means for K = {K}...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_lab


    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                 preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    print(f'K = {K}')
    print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                         labelled_ari))
    print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                           unlabelled_ari))

    return -labelled_acc



@torch.no_grad()
def scipy_optimise(merge_test_loader, model, args):

    small_k = args.num_labeled_classes
    big_k = args.max_classes

    test_k_means_partial = partial(test_kmeans_for_scipy, merge_test_loader=merge_test_loader, model=model, args=args, verbose=True)
    res = minimize_scalar(test_k_means_partial, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    print(f'Optimal K is {res.x}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='aircraft', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--exp_name', default='gcd_estimate_k', type=str)
    parser.add_argument('--max_classes', default=1000, type=int)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()

    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    cluster_accs = {}
    # ----------------------
    # BASE MODEL
    # ----------------------
    args.image_size = 224
    args.interpolation = 3
    args.crop_pct      = 0.875

    # load model from local path

    backbone = vits.__dict__['vit_base']()
    state_dict = torch.load('./pretrained_models/dino/dino_vitbase16_pretrain.pth', map_location='cpu')
    backbone.load_state_dict(state_dict)

    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = int(estimated_k_gb_gcd_and_ours[args.dataset_name])

    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model     = nn.Sequential(backbone, projector)
    args.best_model_path = './dev_outputs/gcd/log/aircraft_simgcd_(21.01.2024_|_02.198)/checkpoints/model_best.pt'
    model     = load_trained_paras(args.best_model_path, [model], ['model'])[0]
    model.to(device)


    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         test_transform,
                                                                                         test_transform,
                                                                                         args)
    
    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # ----------------------
    # estimate k
    # ----------------------
    scipy_optimise(merge_test_loader=train_loader, model=model[0], args=args)
    

 