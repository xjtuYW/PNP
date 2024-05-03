import time
import numpy as np
from tqdm import tqdm
import infomap
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os
import faiss
import math
import pdb


import torch
import torch.nn.functional as F

""""
code from 
"""


min_sim = 0.58

class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None
    

    
def l2norm(vec):
    """
    归一化
    :param vec: 
    :return: 
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb:每一个顶点对应一个类
    lb2idxs:每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):
        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            feats = feats.astype('float32')
            size, dim = feats.shape
            if knn_method == 'faiss-gpu':
                i = math.ceil(size/1000000)
                if i > 1:
                    i = (i-1)*4
                res = faiss.StandardGpuResources()
                res.setTempMemory(i * 1024 * 1024 * 1024)
                index = faiss.GpuIndexFlatIP(res, dim)
            else:
                index = faiss.IndexFlatIP(dim)
            index.add(feats)
        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            sims, nbrs = index.search(feats, k=k)
            # torch.cuda.empty_cache()
            self.knns = [(np.array(nbr, dtype=np.int32),
                            1 - np.array(sim, dtype=np.float32))
                            for nbr, sim in zip(nbrs, sims)]


def knns2ordered_nbrs(knns, sort=False): # sort or not makes no sense
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, tao_f = 0.6):

    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - tao_f:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


# def get_links(single, links, nbrs, dists, tao_f = 0.6):
#     slope = 0.5
#     bias_up  = 1
#     bias_min = tao_f
#     for i in tqdm(range(nbrs.shape[0])):
#         count = 0
#         for j in range(0, len(nbrs[i])):
#             # 排除本身节点
#             dists[i][j] = 0 if 1 - dists[i][j] < 0 else dists[i][j]
#             if i == nbrs[i][j]:
#                 pass
#             elif dists[i][j] <= 1 - tao_f:
#                 count += 1
#                 links[(i, nbrs[i][j])] = float(1 - dists[i][j])
#             else:
#                 count += 1
#                 links[(i, nbrs[i][j])] = (float(1 - dists[i][j])+bias_min) * slope
#         # 统计孤立点
#         if count == 0:
#             single.append(i)
#     return single, links


def get_dist_nbr(features, k=80, knn_method='faiss-cpu'):

    index = knn_faiss(feats=features, k=k, knn_method=knn_method)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs

@torch.no_grad()
def search_with_large_k(features, k=2500):
    features        = torch.FloatTensor(features)
    features        = F.normalize(features, dim=-1)
    ssim            = F.linear(features, features) # [bs, bs]
    values, idxs    = torch.sort(ssim, dim=-1, descending=True)
    values, idxs    = values[:, :k], idxs[:, :k]
    dists           = 1 - np.array(values, dtype=np.float32)
    nbrs            = idxs
    return dists, nbrs


def cluster_by_infomap(features, knn_method='faiss-gpu', tao_f = 0.6, k=10, dataset_name='others'):
    """
    基于infomap的聚类
    @features: np, [-1, 768]
    """

    # get dist and corresponding index
    if dataset_name == 'cifar10':
        dists, nbrs = search_with_large_k(features=features, k=k)
    else:
        dists, nbrs = get_dist_nbr(features=features, k=k, knn_method=knn_method)

    # calc adjcent matrix
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, tao_f=tao_f)

    # build link
    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    ##### build link
    # max_v = max(links.values())
    # min_v = min(links.values())
    # infomapWrapper = infomap.Infomap("--two-level --directed")
    # for (i, j), sim in tqdm(links.items()):
    #     # sim = (sim-min_v) / (max_v-min_v)
    #     sim = (2-sim) / 2
    #     _ = infomapWrapper.addLink(int(i), int(j), sim)
    #####

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)
    
    # preds = []    
    # for i in range(len(idx2label)):
    #     preds.append(idx2label[i])
    # preds = np.array(preds)
    
    return idx2label, single, dists
  

  
# def cluster_by_infomapV2(features, k=1, knn_method='faiss-gpu', epoch=1, tao_f = 0.6):
#     """
#     基于infomap的聚类
#     @features: np, [-1, 768]
#     """
#     sim_matrix   = np.matmul(features, features.transpose(1,0))
#     bs           = sim_matrix.shape[0]

#     pdb.set_trace()
#     links   = {}
#     single  = []
#     for i in range(bs):
#         count = 0
#         for j in range(bs):
#             cur_sim = sim_matrix[i, j]
#             if i == j:
#                 pass
#             elif cur_sim >= tao_f:
#                 count           += 1
#                 links[(i, j)]   = cur_sim
#         if count == 0:
#             single.append(i)

#     # build link
#     infomapWrapper = infomap.Infomap("--two-level --directed")
#     for (i, j), sim in tqdm(links.items()):
#         _ = infomapWrapper.addLink(int(i), int(j), sim)

#     # 聚类运算
#     infomapWrapper.run()

#     label2idx = {}
#     idx2label = {}

#     # 聚类结果统计
#     for node in infomapWrapper.iterTree():
#         # node.physicalId 特征向量的编号
#         # node.moduleIndex() 聚类的编号
#         idx2label[node.physicalId] = node.moduleIndex()
#         if node.moduleIndex() not in label2idx:
#             label2idx[node.moduleIndex()] = []
#         label2idx[node.moduleIndex()].append(node.physicalId)
    
#     # preds = []    
#     # for i in range(len(idx2label)):
#     #     preds.append(idx2label[i])
#     # preds = np.array(preds)
    
#     return idx2label, single


def run_kmeans(x, num_cluster, gpu:int=0, temperature:float=0.2):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}

    # intialize faiss clustering parameters
    d                            = x.shape[1]
    k                            = int(num_cluster)
    clus                         = faiss.Clustering(d, k)
    clus.verbose                 = False
    clus.niter                   = 300
    clus.nredo                   = 5
    clus.seed                    = 666
    clus.max_points_per_centroid = 500
    clus.min_points_per_centroid = 10

    res             = faiss.StandardGpuResources()
    cfg             = faiss.GpuIndexFlatConfig()
    cfg.useFloat16  = False
    cfg.device      = gpu    
    index           = faiss.GpuIndexFlatL2(res, d, cfg)  

    clus.train(x, index)   

    D, I        = index.search(x, 1) # for each sample, find cluster distance and assignments
    im2cluster  = [int(n[0]) for n in I]
    
    # get cluster centroids
    centroids   = faiss.vector_to_array(clus.centroids).reshape(k,d)
    
    # sample-to-centroid distances for each cluster 
    Dcluster    = [[] for c in range(k)]          
    for im,i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])
    
    # concentration estimation (phi)        
    density = np.zeros(k)
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = temperature * density / density.mean()  #scale the mean to temperature 
    
    # convert to cuda Tensors for broadcast
    # centroids   = torch.Tensor(centroids).cuda()
    # centroids   = nn.functional.normalize(centroids, p=2, dim=1)    

    # im2cluster  = torch.LongTensor(im2cluster).cuda()               
    # density     = torch.Tensor(density).cuda()
    
    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)    
        
    return results