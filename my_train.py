import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# export CUDA_VISIBLE_DEVICES= '1'

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
from models.encoder_resnet import  ResNet18

from sklearn.cluster import KMeans
# from finch import FINCH
from cluster import *


### cub : 0.6541	0.6611	0.6507
def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    
    start_epoch = 0
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None

    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                                        optimizer,
                                        T_max=args.epochs,
                                        eta_min=args.lr * 1e-3,
                                    )
    
    ema_scheduler    = EMAScheduler(
                                        w_base=args.TMS_weight_base, 
                                        w_t=0.999, 
                                        max_epoch=200
                                    )
    
    best_train_acc_all = 0

    if args.resume:
        device              = torch.device('cuda:0')
        checkpoints         = torch.load(args.model_path, map_location=device)
        best_train_acc_all  = checkpoints['best_acc']
        start_epoch         = checkpoints['epoch']
        student.load_state_dict(checkpoints['model'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoints['lr_schedule'])
        ema_scheduler.load_state_dict(checkpoints['ema_scheduler'])

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # initialize
    if args.encoder_arch == 'dino':
        teacher       = nn.Sequential(vits.__dict__['vit_base'](), 
                                    DINOHeadV2(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, enable_virtual_cls=args.enable_virtual_cls)).cuda()
    elif args.encoder_arch == 'resnet18':
        teacher       = nn.Sequential(ResNet18(), 
                                    DINOHeadV2(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, enable_virtual_cls=args.enable_virtual_cls)).cuda()
    
    for param_stu, param_tea in zip(student.parameters(), teacher.parameters()):
            param_tea.data.copy_(param_stu.data)  # initialize
            param_tea.requires_grad = False  # not update by gradient
    teacher.eval()


    for epoch in range(start_epoch, args.epochs):
        loss_record = AverageMeter()
        student.train()

        if args.enable_TMS:
            args.ema_param = ema_scheduler(epoch)
        else:
            args.ema_param = 0

        args.logger.info("Epoch:{} \t ema_weights:{:.4f}".format(epoch, args.ema_param))
      
        for batch_idx, batch in enumerate(train_loader):
            """
            images: List, [[bs, C, H, W], ..., [bs, C, H, W]]
            class_labels: Tensor,  [bs]
            uq_idxs: Tensor,  [bs]
            mask_lab: Tensor,  [bs] 0-1
            """
            proto_mem = student[1].prototypes.data

            images, class_labels, uq_idxs, mask_lab = batch
           
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images                 = torch.cat(images, dim=0).cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
     
                student_proj, student_out, student_out_unl = student(images) # student_proj:[n_views*bs, proj_dim]
                with torch.no_grad():
                    teacher_proj, teacher_out, teacher_out_unl = teacher(images) # [n_views*bs, num_cls]

                # clustering, sup
                sup_logits  = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0) # select data with label
                sup_labels  = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss    = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    
                # clustering, unsup
                cluster_loss    = cluster_criterion(student_out_unl, teacher_out_unl, epoch)
                avg_probs       = (student_out_unl / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss     = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss    += args.memax_weight * me_max_loss

                # represent learning, unsup 
                if args.enable_DInfoNCE:
                    contrastive_logits_list, contrastive_labels_list = info_nce_logitsV2(student_proj, teacher_proj)
                    contrastive_loss, n_loss_terms = 0, 0
                    for contrastive_logits, contrastive_labels in zip(contrastive_logits_list, contrastive_labels_list):
                        contrastive_loss    += torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
                        n_loss_terms        += 1
                    contrastive_loss = contrastive_loss / n_loss_terms
                else:
                    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
  
                # representation learning, sup
                student_proj    = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj    = torch.nn.functional.normalize(student_proj, dim=-1, p=2)
                sup_con_labels  = class_labels[mask_lab]
                sup_con_loss    = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'me_max_loss: {me_max_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                if args.loss_ablat == 'wo_cls':
                    loss += cluster_loss + (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                elif args.loss_ablat == 'wo_st':
                    loss += cls_loss + (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                elif args.loss_ablat == 'wo_sup':
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss + sup_con_loss
                elif args.loss_ablat == 'wo_unsup':
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss + contrastive_loss
                else:
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
            
            # EMA
            with torch.no_grad():
                student[1].prototypes.data = proto_mem * args.ema_param + student[1].prototypes.data * (1-args.ema_param)
                for param_stu, param_tea in zip(student[0].parameters(), teacher[0].parameters()):
                    param_tea.data = param_tea.data * args.ema_param + param_stu.data * (1-args.ema_param)
                for param_stu, param_tea in zip(student[1].parameters(), teacher[1].parameters()):
                    param_tea.data = param_stu.data

        # Step schedule
        exp_lr_scheduler.step()

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        if (epoch + 1) % 1 == 0:
            args.logger.info('Testing on unlabelled examples in the training data...')

            all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
           
            # when using prototypes to update the model's head, we also update the teacher's head
            with torch.no_grad():
                teacher[1].act_protos = student[1].act_protos
                for param_stu, param_tea in zip(student[1].parameters(), teacher[1].parameters()):
                    param_tea.data = param_stu.data

            # args.logger.info('Testing on disjoint test set...')
            # all_acc_test, old_acc_test, new_acc_test = testV1(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            # save best model
            if best_train_acc_all < all_acc:
                best_train_acc_all = all_acc

                save_dict = {'model': student.state_dict()}

                torch.save(save_dict, args.best_model_path)
                args.logger.info("model saved to {}.".format(args.best_model_path))
                args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        # save checkpoint
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule':exp_lr_scheduler.state_dict(),
            'ema_scheduler':ema_scheduler.state_dict(),
            'epoch': epoch + 1,
            'best_acc': best_train_acc_all,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model checkpoint is saved to {}.".format(args.model_path))


@torch.no_grad()
def match_idx(protos, weights):
    p_numpy = protos
    w_numpy = weights.clone().detach().cpu().numpy()
    sims    = np_cosine_sim(p_numpy, w_numpy)
    ind = linear_assignment(sims.max(axis=-1, keepdims=True) - sims)
    ind = np.vstack(ind).T

    return ind[:,1]
    
@torch.no_grad()
def test(model, test_loader, epoch, save_name, args, reload:bool=False):
  
    if reload:
        model = load_trained_paras(args.best_model_path, [model], ['model'])[0]
  
    model.eval()

    all_feats_norm, all_feats_unnorm = [], []
    targets     = np.array([])
    all_preds   = []
    mask        = np.array([])

    # timer = Timer()

    print('Collecting features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
  
        images            = images.cuda(non_blocking=True)

        # forward
        _, _, logits      = model(images)
        feats_unmorm      = model[0].feat_tmp
        feats_norm        = torch.nn.functional.normalize(feats_unmorm, dim=-1, p=2)

        # collect normalized and original features
        all_feats_unnorm.append(feats_unmorm.cpu().numpy())
        all_feats_norm.append(feats_norm.cpu().numpy())

        # collect predictions and gt
        all_preds.append(logits.argmax(1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask    = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
 
    # -----------------------
    # Infomap
    # -----------------------
    print('Fitting Infomaps...')
    all_feats_unnorm             = np.concatenate(all_feats_unnorm) 
    all_feats_norm               = np.concatenate(all_feats_norm)
    cluster_target               = all_feats_norm
    idx2label, idx_single, dists = cluster_by_infomap(cluster_target, k=args.top_k, tao_f=args.tao_f, dataset_name=args.dataset_name)
    
    # get centroid using the idx2label
    preds_unsingle          = []  
    idxs_unsingle           = [] 
    for i in idx2label.keys():
        preds_unsingle.append(idx2label[i])
        idxs_unsingle.append(i)
    preds_unsingle      = np.array(preds_unsingle)
    preds_unique_label  = np.unique(preds_unsingle)
    feats_unsingle      = cluster_target[idxs_unsingle]
    centroid            = []
    for cls_id in preds_unique_label:
        data_idx    = np.where(preds_unsingle==cls_id)[0]
        data        = feats_unsingle[data_idx] # [n, d]
        prototype   = data.mean(axis=0, keepdims=True)
        centroid.append(prototype)

    # use the obtained centroid to predict the lable to single data
    centroid = np.concatenate(centroid) # [-1, 768]

    # update the unlablled head
    estimate_k  = centroid.shape[0]
    args.logger.info(f'estimate_k is {estimate_k}')
    model[1].act_protos              = estimate_k
    model[1].prototypes[:estimate_k] = torch.Tensor(centroid).cuda() 

    print('Done!')

    # predict the single data using obtained centroid 
    if len(idx_single) != 0:
        single_feats  = all_feats_norm[idx_single] 
        single_logits = np_cosine_sim(single_feats, centroid)
        single_preds  = single_logits.argmax(1)
    preds = []
    single_cnt = 0 
    for i in range(all_feats_norm.shape[0]):
        if i in idx2label:
            preds.append(idx2label[i])
        else:
            if len(idx_single) != 0:
                preds.append(single_preds[single_cnt])
                single_cnt += 1
            else:
                preds.append(all_preds[i])

    preds                     = np.array(preds)
   
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


def test_kmeans(model, test_loader, epoch, save_name, args, reload:bool=False):

    args.best_model_path = './dev_outputs/simgcd/log/best/checkpoints/model_best.pt'
    # pdb.set_trace()
    if reload:
        model = load_trained_paras(args.best_model_path, [model], ['model'])[0]

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        _ = model(images)
        feats = model[0].feat_tmp

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)

    return all_acc, old_acc, new_acc


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
# enable_virtual_cls
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--resume', action='store_true', default=False)
    
    parser.add_argument('--enable_TMS', type=str2bool, default=True)
    parser.add_argument('--enable_virtual_cls', type=str2bool, default=True)
    parser.add_argument('--TMS_weight_base', type=float, default=0.7)
    parser.add_argument('--enable_DInfoNCE', type=str2bool, default=True)
    parser.add_argument('--tao_f', type=float, default=0.6)
    parser.add_argument('--vp_num', type=float, default=2)
    parser.add_argument('--loss_ablat', type=str, default='w_all')

    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--method_version', default='V2', type=str)
    parser.add_argument('--ema_param', type=float, default=0.9)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--encoder_arch', type=str, default='dino')
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    set_seed(666)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    resume_path = None
    # resume_path = 'dev_outputs/NPC/log/scars_npc_(06.03.2024_|_24.118)'
    init_experiment(args, runner_name=['NPC'], resume_path=resume_path)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct      = 0.875

    # load model from local path
    if args.encoder_arch == 'dino':
        args.logger.info('Backbone using vit pretrained with dino')
        backbone = vits.__dict__['vit_base']()
        state_dict = torch.load('./pretrained_models/dino/dino_vitbase16_pretrain.pth', map_location='cpu')
        backbone.load_state_dict(state_dict)

        # load model from remote services
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        if args.warmup_model_dir is not None:
            args.logger.info(f'Loading weights from {args.warmup_model_dir}')
            backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
            
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

        args.feat_dim = 768
        args.num_mlp_layers = 3
    
    elif args.encoder_arch == 'resnet18':
        args.logger.info('Backbone using resnet18')
        backbone      = ResNet18()
        args.feat_dim = 512
        args.num_mlp_layers = 1
    
    args.logger.info('model build')

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size     = 224
    args.mlp_out_dim    = int((args.num_labeled_classes + args.num_unlabeled_classes)*args.vp_num)

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len       = len(train_dataset.labelled_dataset)
    unlabelled_len  = len(train_dataset.unlabelled_dataset)
    sample_weights  = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights  = torch.DoubleTensor(sample_weights)
    sampler         = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector   = DINOHeadV2(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, enable_virtual_cls=args.enable_virtual_cls)
    model       = nn.Sequential(backbone, projector).to(device)
   
    # ----------------------
    # TRAIN
    # ----------------------
    print('TRAIN...')
    train(model, train_loader, None, test_loader_unlabelled, args)

    # ----------------------
    # TEST
    # ----------------------
    # 
    print('TEST...')
    all_acc, old_acc, new_acc = test(model, test_loader_unlabelled, epoch=-1, save_name='Train ACC Unlabelled', args=args, reload=True)

    # all_acc, old_acc, new_acc = test_kmeans(model, test_loader_unlabelled, epoch=-1, save_name='Train ACC Unlabelled', args=args, reload=True)
    args.logger.info('Best Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
