import os
import torch
import inspect
import re
import numpy as np

from datetime import datetime
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

import torch.distributed as dist
import math
from PIL import ImageFilter
import time

# calculate consumption time
class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(x / 60)
        return '{}s'.format(x)

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None, resume_path=None):
    if resume_path is not None:
        log_dir     = resume_path
        args.resume = True
    else:
        # Get filepath of calling script
        if runner_name is None:
            runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

        root_dir = os.path.join(args.exp_root, *runner_name)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Either generate a unique experiment ID, or use one which is passed
        if exp_id is None:

            if args.exp_name is None:
                raise ValueError("Need to specify the experiment name")
            # Unique identifier for experiment
            now = '{}_({:02d}.{:02d}.{}_|_'.format(args.exp_name, datetime.now().day, datetime.now().month, datetime.now().year) + \
                datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)
            while os.path.exists(log_dir):
                now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                    datetime.now().strftime("%S.%f")[:-3] + ')'

                log_dir = os.path.join(root_dir, 'log', now)

        else:

            log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')
    args.best_model_path = os.path.join(args.model_dir, 'model_best.pt')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    print(runner_name)
    print(args)

    return args


class DistributedWeightedSamplerV2(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples



def load_trained_paras(path: str, models: list, keys:list, map_location="cpu", logger=None, sub_level=None):
    if logger:
        logger.info(f"Load pretrained model [{model.__class__.__name__}] from {path}")
    if os.path.exists(path):
        # From local
        state_dict = torch.load(path, map_location)
    # elif path.startswith("http"):
    #     # From url
    #     state_dict = load_state_dict_from_url(path, map_location=map_location, check_hash=False)
    else:
        raise Exception(f"Cannot find {path} when load pretrained")
    
    model_trained = []
    for i in range(len(models)):
        model, key = models[i], keys[i]
        model = load_pretrained_dict(model, state_dict, key)
        model_trained.append(model)

    return model_trained


def _auto_drop_invalid(model: torch.nn.Module, state_dict: dict, logger=None):
    """ Strip unmatched parameters in state_dict, e.g. shape not matched, type not matched.

    Args:
        model (torch.nn.Module):
        state_dict (dict):
        logger (logging.Logger, None):

    Returns:
        A new state dict.
    """
    ret_dict = state_dict.copy()
    invalid_msgs = []
    for key, value in model.state_dict().items():
        if key in state_dict:
            # Check shape
            new_value = state_dict[key]
            if value.shape != new_value.shape:
                invalid_msgs.append(f"{key}: invalid shape, dst {value.shape} vs. src {new_value.shape}")
                ret_dict.pop(key)
            elif value.dtype != new_value.dtype:
                invalid_msgs.append(f"{key}: invalid dtype, dst {value.dtype} vs. src {new_value.dtype}")
                ret_dict.pop(key)
    if len(invalid_msgs) > 0:
        warning_msg = "ignore keys from source: \n" + "\n".join(invalid_msgs)
        if logger:
            logger.warning(warning_msg)
        else:
            import warnings
            warnings.warn(warning_msg)
    return ret_dict


def load_pretrained_dict(model: torch.nn.Module, state_dict: dict, key:str, logger=None, sub_level=None):
    """ Load parameters to model with
    1. Sub name by revise_keys For DataParallelModel or DistributeParallelModel.
    2. Load 'state_dict' again if possible by key 'state_dict' or 'model_state'.
    3. Take sub level keys from source, e.g. load 'backbone' part from a classifier into a backbone model.
    4. Auto remove invalid parameters from source.
    5. Log or warning if unexpected key exists or key misses.

    Args:
        model (torch.nn.Module):
        state_dict (dict): dict of parameters
        logger (logging.Logger, None):
        sub_level (str, optional): If not None, parameters with key startswith sub_level will remove the prefix
            to fit actual model keys. This action happens if user want to load sub module parameters
            into a sub module model.
    """
    revise_keys = [(r'^module\.', '')]
    state_dict = state_dict[key]
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    if sub_level:
        sub_level = sub_level if sub_level.endswith(".") else (sub_level + ".")
        sub_level_len = len(sub_level)
        state_dict = {key[sub_level_len:]: value
                      for key, value in state_dict.items()
                      if key.startswith(sub_level)}

    state_dict = _auto_drop_invalid(model, state_dict, logger=logger)

    load_status = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = load_status.unexpected_keys
    missing_keys = load_status.missing_keys
    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msgs.append('missing key in source '
                        f'state_dict: {", ".join(missing_keys)}\n')
    err_msgs = '\n'.join(err_msgs)

    if len(err_msgs) > 0:
        if logger:
            logger.warning(err_msgs)
        else:
            import warnings
            warnings.warn(err_msgs)
    return model



# class GradualWarmupScheduler(_LRScheduler):
#     """ Gradually warm-up(increasing) learning rate in optimizer.
#     Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
#         total_epoch: target learning rate is reached at total_epoch, gradually
#         after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
#     """

#     def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
#         self.multiplier         = multiplier
#         if self.multiplier < 1.:
#             raise ValueError('multiplier should be greater thant or equal to 1.')
#         self.total_epoch = total_epoch
#         self.after_scheduler = after_scheduler
#         self.finished = False
#         super(GradualWarmupScheduler, self).__init__(optimizer)

#     def get_lr(self):
#         if self.last_epoch > self.total_epoch:
#             if self.after_scheduler:
#                 if not self.finished:
#                     self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
#                     self.finished = True
#                 return self.after_scheduler.get_last_lr()
#             return [base_lr * self.multiplier for base_lr in self.base_lrs]

#         if self.multiplier == 1.0:
#             return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
#         else:
#             return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
   
#     def setp_CosineAnnealingLR(self, metrics, epoch=None):
#         pass

#     def step_ReduceLROnPlateau(self, metrics, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
#         if self.last_epoch <= self.total_epoch:
#             warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
#             for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
#                 param_group['lr'] = lr
#         else:
#             if epoch is None:
#                 self.after_scheduler.step(metrics, None)
#             else:
#                 self.after_scheduler.step(metrics, epoch - self.total_epoch)

#     def step(self, epoch=None, metrics=None):
#         if type(self.after_scheduler) != ReduceLROnPlateau:
#             if self.finished and self.after_scheduler:
#                 if epoch is None:
#                     self.after_scheduler.step(None)
#                 else:
#                     self.after_scheduler.step(epoch - self.total_epoch)
#                 self._last_lr = self.after_scheduler.get_last_lr()
#             else:
#                 return super(GradualWarmupScheduler, self).step(epoch)
#         else:
#             self.step_ReduceLROnPlateau(metrics, epoch)


class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def np_cosine_sim(feat1, feat2):
    feat1 = feat1 / np.linalg.norm(feat1, axis=1).reshape(-1, 1)
    feat2 = feat2 / np.linalg.norm(feat2, axis=1).reshape(-1, 1)
    return np.matmul(feat1, feat2.transpose(1,0))


def np_softmax(x):
    """ softmax function """
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    return x
