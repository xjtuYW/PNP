import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
from scipy.optimize import linear_sum_assignment as linear_assignment
from util.general_utils import GaussianBlur
from torchvision import transforms

import pdb

class CLSHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True):
        super().__init__()
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits     = self.last_layer(x)
        return logits


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, args, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, 5*out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits_unlabeled    = self.last_layer(x)
        return x_proj, logits_unlabeled


class DINOHeadV2(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, enable_virtual_cls=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        
        # prototypes for unlabelled data
        self.prototypes = nn.Parameter(torch.randn((out_dim, in_dim), requires_grad=True))
        # self.prototypes = nn.Parameter(torch.randn((int(out_dim*4), in_dim), requires_grad=True))
        self.act_protos = out_dim

        self.enable_virtual_cls = enable_virtual_cls


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj              = self.mlp(x)
        x                   = nn.functional.normalize(x, dim=-1, p=2)
        logits_labelled     = self.last_layer(x)
        logits_unlabeled    = F.linear(x, nn.functional.normalize(self.prototypes, dim=-1, p=2))

        if self.enable_virtual_cls:
            return x_proj, logits_labelled, logits_unlabeled
        else:
            return x_proj, logits_labelled, logits_unlabeled[:,:self.act_protos]


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    
    # def multi_transform(self,x):
        
    #     if not isinstance(self.base_transform, list):
    #         return [self.base_transform(x) for i in range(self.n_views)]
    #     else:
    #         return [self.base_transform[i](x) for i in range(self.n_views)]

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


class ContrastiveLearningViewGeneratorV2(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2, aug_method='MoCoV2'):
        self.weak_aug = base_transform
        self.n_views = n_views

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        if aug_method=='MoCoV2':
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            self.strong_aug = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

        elif aug_method=='MoCoV1':
            # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
            self.strong_aug = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif aug_method=='muGCD':
            pass
            # self.strong_aug = transforms.Compose([
            #     transforms.RandomResizedCrop(224, scale=(0.3, 1.)),
            #     transforms.RandomApply([
            #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #     ], p=0.8),
            #     transforms.RandomGrayscale(p=0.2),
            #     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])

        if n_views == 2:
            self.transforms = [self.weak_aug, self.strong_aug]
        elif n_views == 3:
            self.transforms = [self.weak_aug, self.weak_aug, self.strong_aug]
        elif n_views == 4:
            # self.transforms = [self.weak_aug, self.weak_aug, self.weak_aug, self.strong_aug]
            self.transforms = [self.weak_aug, self.weak_aug, self.weak_aug, self.weak_aug]

    def __call__(self, x):
        if not isinstance(self.transforms, list):
            return [self.transforms(x) for i in range(self.n_views)]
        else:
            return [self.transforms[i](x) for i in range(self.n_views)]


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def info_nce_logitsV2(features_stu, features_tea, n_views=2, temperature=1.0, device='cuda'):
    
    logits_list, lables_list = [], []
    if n_views == 1:
        feature = torch.cat((features_stu, features_tea), dim=0)
        logits, labels = info_nce_logits(feature, n_views=2, temperature=temperature, device=device)
        logits_list.append(logits)
        lables_list.append(labels)

    else:
        stu_proj = features_stu.chunk(2)
        tea_proj = features_tea.chunk(2)
        for iq, q in enumerate(stu_proj):
            for v in range(len(tea_proj)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
            feature = torch.cat((q, tea_proj[v]), dim=0)
            logits, labels = info_nce_logits(feature, n_views=n_views, temperature=temperature, device=device)
            logits_list.append(logits)
            lables_list.append(labels)

    return logits_list, lables_list


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class DistillLossV2(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        if self.ncrops == 1:
            student_out = student_output / self.student_temp
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax(teacher_output / temp, dim=-1)
            total_loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        else:
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.ncrops)

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax(teacher_output / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(2)

            total_loss = 0
            n_loss_terms = 0
            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
            total_loss /= n_loss_terms
        return total_loss

class EMAScheduler(nn.Module):
    """"
    Following muGCD to implement this, https://arxiv.org/pdf/2311.17055.pdf
    """
    def __init__(self, w_base=0.7, w_t=0.999, max_epoch=200):
        super().__init__()
        self.w_base     = w_base
        self.w_t        = w_t
        self.max_epoch  = max_epoch
    
    def forward(self, cur_epoch):
        if cur_epoch > self.max_epoch:
            return self.w_t
        return self.w_t - (1-self.w_base)*(math.cos((math.pi*cur_epoch)/self.max_epoch)+1)/2

