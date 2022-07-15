import h5py
import pandas as pd
import json
import numpy as np
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ssp
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import pickle
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix
from collections import Counter
import pickle
from sklearn.utils import shuffle
from os.path import join as pjoin
from tqdm.auto import tqdm
import os
import joblib
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import importlib
from sklearn.preprocessing import normalize as sk_normalize
import pytorch_lightning as pl
from os.path import join as pjoin
from scipy import sparse

def s2t(data):
    try:
        return torch.FloatTensor(data.toarray())
    except:
        return torch.FloatTensor(data)


class dictDataset(Dataset):

    def __init__(self, masks, matrices, categoricals, tr_scalar, targets, decay_rate=0.95):
        super(dictDataset, self).__init__()
        self.targets = targets
        self.tr_scalar = tr_scalar.astype(np.float32)
        self.n_users = len(targets)
        self.matrices = matrices
        self.decay_rate = decay_rate
        self.categoricals = categoricals
        self.masks = masks

    def __getitem__(self, i):
        mask = s2t(self.masks[i]).squeeze(0)
        target = self.targets[i]
        ret_scalar = self.tr_scalar[i]
        ret_mat = {}
        for mat_id in self.matrices:
            ret_mat[mat_id] = s2t(self.matrices[mat_id][i]).squeeze(0)
        ret_cat = {}
        for cat_id in self.categoricals:
            ret_cat[cat_id] = self.categoricals[cat_id][0][i]
        return (target, mask), (ret_mat, ret_cat, ret_scalar)


    def __len__(self,):
        return self.n_users

def validate(model, val_ds, use_filter=True, skip_val_only=False):
    hr = []
    model = model.eval()
    s = 0
    tot_mrr = []
    ii = 0
    for input in DataLoader(val_ds, batch_size=250, shuffle=True):
        (target_idx, _), (ret_mat, ret_cat, ret_scalar) = input
        for k in ret_mat:
            ret_mat[k] = ret_mat[k].cuda()
        for k in ret_cat:
            ret_cat[k] = ret_cat[k].cuda()
        ret_scalar = ret_scalar.cuda()
        ret = model.forward(ret_mat, ret_cat, ret_scalar,)
        top_rec = model.get_top_rec((ret_mat, ret_cat, ret_scalar), use_filter=use_filter)

#         if skip_val_only:
#         top_rec = top_rec[target_idx != 29999]
#         target_idx = target_idx[target_idx != 29999]

        mrr = (top_rec == target_idx.unsqueeze(1)).float().numpy()
#         mrr = mrr.sum(-1)
        mrr = (mrr / np.expand_dims(np.arange(1, 1 + 100), 0)).sum(-1)
        tot_mrr.extend(mrr.tolist())
    model = model.train()
    return np.mean(tot_mrr)



def flatten(t):
    return [item for sublist in t for item in sublist]

class MLP(pl.LightningModule):
    def __init__(self, tr_matrices, tr_categoricals, tr_scalar, n_items,
                 dim=256, layer_dim=1024, dropout=0.1, lr=1e-3, use_sc=True, use_cat=False, light=False,
                num_feats=903):
        super(MLP, self).__init__()

        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items, dim)
        self.feat_emb = nn.Embedding(num_feats, dim)
        self.scalar_emb = nn.Linear(tr_scalar.shape[-1], 64)

        self.use_sc = use_sc
        self.use_cat = use_cat

        self.scalar_bn = nn.BatchNorm1d(64)
        self.lr = lr
        
        if light is True:
            self.item_mat_key = []
        else:
            self.item_mat_key = ['item_last_20', 'item_last_5', 'item_last_2', 'item_last_1']
            
        self.feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']
        
        dim = dim

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = 0

        if self.use_sc:
            aggr_width += 64

        for mat_id, mat in tr_matrices.items():
            if mat_id not in self.feat_mat_key + self.item_mat_key:
                continue 
            print(mat_id)
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(dim)])
            aggr_width += dim

        if self.use_cat:
            for cat_id, cat in tr_categoricals.items():
                data, (n_cats, embedding_dim) = cat
                cat_modules[cat_id] = nn.Embedding(n_cats, embedding_dim, padding_idx=-1)
                cat_bns[cat_id] = nn.BatchNorm1d(embedding_dim)
                aggr_width += embedding_dim

        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_layers = nn.ModuleDict(modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)
        self.last_layers = nn.ModuleList([nn.Linear(aggr_width, layer_dim),
                                          nn.Linear(layer_dim, dim)])

        # torch.nn.Identity()
        self.last_bns = nn.ModuleList([nn.BatchNorm1d(layer_dim),
                                       nn.BatchNorm1d(dim)])

#         self.last_proj = nn.Linear(dim, self.n_items)

        self.item_bias = nn.Parameter(torch.zeros(n_items).float(), requires_grad=True)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, mats, cats, scalar_feats, norm_user=False):
        item_embs = self.item_emb.weight
        feat_embs = self.feat_emb.weight

        aggr = []
        if self.use_sc:
            h = self.scalar_bn(self.scalar_emb(scalar_feats))
            aggr.append(h)

#         item_mat_key = ['item_bow', 'item_last_5', 'item_last_2', 'item_last_1']
#         feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']

        for mat_key in self.item_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, item_embs.T)))
            aggr.append(h)

        for mat_key in self.feat_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, feat_embs.T)))
            aggr.append(h)

        
#         for key, input in mats.items():
#             if key in item_mat_key:
#                 continue
#             elif key in feat_mat_key:
#                 continue

#             h = input
#             for l, bn in zip(self.all_layers[key], self.all_bns[key]):
#                 h = torch.tanh(self.drop(l(h)))
#                 h = bn(h)
#             aggr.append(h)

        if self.use_cat:
            for key, input in cats.items():
                h = self.drop(self.cat_embs[key](input))
                h = self.cat_bns[key](h)
                aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        for l, bn in zip(self.last_layers, self.last_bns):
            h = bn(torch.tanh(self.drop(l(h))))

        if norm_user:
            h = F.normalize(h)

        ret = F.linear(h, F.normalize(self.item_emb.weight))
        return ret

    def get_top_rec(self, inputs, K=100, actual_cand=None, use_filter=True):
        ret = self.forward(*inputs)
        if use_filter:
            user_history = inputs[0]['item_bow'].bool()
            ret[user_history] = -10000.0

        top_items = (-ret).argsort()[:, :K]
        return top_items.detach().cpu()

    def init_weights(self):
        self.item_emb.weight.data.normal_(0, 0.001)
        self.feat_emb.weight.data.normal_(0, 0.001)

        torch.clamp(self.feat_emb.weight.data, min=-0.01, max=0.01)
        torch.clamp(self.item_emb.weight.data, min=-0.01, max=0.01)

        for key in self.all_layers:
            try:
                for layer in self.all_layers[key]:
                    # Xavier Initialization for weights
                    layer.weight.data.normal_(0, 0.001)
                    layer.bias.data.normal_(0.0, 0.001)
                    torch.clamp(layer.bias.data, min=-0.01, max=0.01)
            except:
                pass

class MLP2(pl.LightningModule):
    def __init__(self, lm_feats, tr_matrices, tr_categoricals, tr_scalar, n_items,
                 dim=256, layer_dim=1024, dropout=0.1, lr=1e-3, use_sc=True, use_cat=False, light=False):
        super(MLP2, self).__init__()
        
        self.lm_feats = nn.Parameter(torch.FloatTensor(lm_feats), requires_grad=False)
        self.lm_bn = nn.BatchNorm1d(self.lm_feats.shape[1])
        self.lm_feat_proj = nn.Linear(self.lm_feats.shape[1], dim)
        
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items, dim)
        self.feat_emb = nn.Embedding(self.lm_feats.shape[1], dim)
        self.scalar_emb = nn.Linear(tr_scalar.shape[-1], 64)

        self.use_sc = use_sc
        self.use_cat = use_cat

        self.scalar_bn = nn.BatchNorm1d(64)
        self.lr = lr
        
        if light is True:
            self.item_mat_key = []
        else:
            self.item_mat_key = ['item_last_20', 'item_last_5', 'item_last_2', 'item_last_1']
            
        self.feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']
        
        dim = dim

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = 0

        if self.use_sc:
            aggr_width += 64

        for mat_id, mat in tr_matrices.items():
            if mat_id not in self.feat_mat_key + self.item_mat_key:
                continue 
            print(mat_id)
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(dim)])
            aggr_width += dim

        if self.use_cat:
            for cat_id, cat in tr_categoricals.items():
                data, (n_cats, embedding_dim) = cat
                cat_modules[cat_id] = nn.Embedding(n_cats, embedding_dim, padding_idx=-1)
                cat_bns[cat_id] = nn.BatchNorm1d(embedding_dim)
                aggr_width += embedding_dim

        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_layers = nn.ModuleDict(modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)
        self.last_layers = nn.ModuleList([nn.Linear(aggr_width, layer_dim),
                                          nn.Linear(layer_dim, dim)])

        # torch.nn.Identity()
        self.last_bns = nn.ModuleList([nn.BatchNorm1d(layer_dim),
                                       nn.BatchNorm1d(dim)])

#         self.last_proj = nn.Linear(dim, self.n_items)

        self.item_bias = nn.Parameter(torch.zeros(n_items).float(), requires_grad=True)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, mats, cats, scalar_feats, norm_user=False):
        lm_fow = torch.tanh(self.lm_feat_proj(self.lm_bn(self.lm_feats)))
        
        item_embs = self.item_emb.weight
        feat_embs = self.feat_emb.weight
        
        item_embs = lm_fow + item_embs
        
        aggr = []
        if self.use_sc:
            h = self.scalar_bn(self.scalar_emb(scalar_feats))
            aggr.append(h)

#         item_mat_key = ['item_bow', 'item_last_5', 'item_last_2', 'item_last_1']
#         feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']

        for mat_key in self.item_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, item_embs.T)))
            aggr.append(h)

        for mat_key in self.feat_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, feat_embs.T)))
            aggr.append(h)

        
#         for key, input in mats.items():
#             if key in item_mat_key:
#                 continue
#             elif key in feat_mat_key:
#                 continue

#             h = input
#             for l, bn in zip(self.all_layers[key], self.all_bns[key]):
#                 h = torch.tanh(self.drop(l(h)))
#                 h = bn(h)
#             aggr.append(h)

        if self.use_cat:
            for key, input in cats.items():
                h = self.drop(self.cat_embs[key](input))
                h = self.cat_bns[key](h)
                aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        for l, bn in zip(self.last_layers, self.last_bns):
            h = bn(torch.tanh(self.drop(l(h))))

        if norm_user:
            h = F.normalize(h)

        ret = F.linear(h, F.normalize(item_embs))
        return ret

    def get_top_rec(self, inputs, K=100, actual_cand=None, use_filter=True):
        ret = self.forward(*inputs)
        if use_filter:
            user_history = inputs[0]['item_bow'].bool()
            ret[user_history] = -10000.0

        top_items = (-ret).argsort()[:, :K]
        return top_items.detach().cpu()

    def init_weights(self):
        self.item_emb.weight.data.normal_(0, 0.001)
        self.feat_emb.weight.data.normal_(0, 0.001)

        torch.clamp(self.feat_emb.weight.data, min=-0.01, max=0.01)
        torch.clamp(self.item_emb.weight.data, min=-0.01, max=0.01)

        for key in self.all_layers:
            try:
                for layer in self.all_layers[key]:
                    # Xavier Initialization for weights
                    layer.weight.data.normal_(0, 0.001)
                    layer.bias.data.normal_(0.0, 0.001)
                    torch.clamp(layer.bias.data, min=-0.01, max=0.01)
            except:
                pass


class MLP_FEATONLY(pl.LightningModule):
    def __init__(self, lm_feats, tr_matrices, tr_categoricals, tr_scalar, n_items,
                 dim=256, layer_dim=1024, dropout=0.1, lr=1e-3, use_sc=True, use_cat=False, light=False):
        super(MLP_FEATONLY, self).__init__()
        self.lm_feats = nn.Parameter(torch.FloatTensor(lm_feats), requires_grad=False)
        self.lm_bn = nn.BatchNorm1d(self.lm_feats.shape[1])
        self.lm_feat_proj = nn.Linear(self.lm_feats.shape[1], dim)
        
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items, dim)
        self.feat_emb = nn.Embedding(782, dim)
        self.scalar_emb = nn.Linear(tr_scalar.shape[-1], 64)

        self.use_sc = use_sc
        self.use_cat = use_cat

        self.scalar_bn = nn.BatchNorm1d(64)
        self.lr = lr
        
        if light is True:
            self.item_mat_key = []
        else:
            self.item_mat_key = ['item_last_20', 'item_last_5', 'item_last_2', 'item_last_1']
            
        self.feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']
        
        dim = dim

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = 0

        if self.use_sc:
            aggr_width += 64

        for mat_id, mat in tr_matrices.items():
            if mat_id not in self.feat_mat_key + self.item_mat_key:
                continue 
            print(mat_id)
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(dim)])
            aggr_width += dim

        if self.use_cat:
            for cat_id, cat in tr_categoricals.items():
                data, (n_cats, embedding_dim) = cat
                cat_modules[cat_id] = nn.Embedding(n_cats, embedding_dim, padding_idx=-1)
                cat_bns[cat_id] = nn.BatchNorm1d(embedding_dim)
                aggr_width += embedding_dim

        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_layers = nn.ModuleDict(modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)
        self.last_layers = nn.ModuleList([nn.Linear(aggr_width, layer_dim),
                                          nn.Linear(layer_dim, dim)])

        # torch.nn.Identity()
        self.last_bns = nn.ModuleList([nn.BatchNorm1d(layer_dim),
                                       nn.BatchNorm1d(dim)])

#         self.last_proj = nn.Linear(dim, self.n_items)

        self.item_bias = nn.Parameter(torch.zeros(n_items).float(), requires_grad=True)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, mats, cats, scalar_feats, norm_user=False):
        
        item_embs = torch.tanh(self.lm_feat_proj(self.lm_bn(self.lm_feats)))
        
        feat_embs = self.feat_emb.weight

        aggr = []
        if self.use_sc:
            h = self.scalar_bn(self.scalar_emb(scalar_feats))
            aggr.append(h)

#         item_mat_key = ['item_bow', 'item_last_5', 'item_last_2', 'item_last_1']
#         feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']

        for mat_key in self.item_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, self.item_emb.weight.T)))
            aggr.append(h)

        for mat_key in self.feat_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, feat_embs.T)))
            aggr.append(h)

        
#         for key, input in mats.items():
#             if key in item_mat_key:
#                 continue
#             elif key in feat_mat_key:
#                 continue

#             h = input
#             for l, bn in zip(self.all_layers[key], self.all_bns[key]):
#                 h = torch.tanh(self.drop(l(h)))
#                 h = bn(h)
#             aggr.append(h)

        if self.use_cat:
            for key, input in cats.items():
                h = self.drop(self.cat_embs[key](input))
                h = self.cat_bns[key](h)
                aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        for l, bn in zip(self.last_layers, self.last_bns):
            h = bn(torch.tanh(self.drop(l(h))))

        if norm_user:
            h = F.normalize(h)

        ret = F.linear(h, F.normalize(item_embs))
        return ret

    def get_top_rec(self, inputs, K=100, actual_cand=None, use_filter=True):
        ret = self.forward(*inputs)
        if use_filter:
            user_history = inputs[0]['item_bow'].bool()
            ret[user_history] = -10000.0

        top_items = (-ret).argsort()[:, :K]
        return top_items.detach().cpu()

    def init_weights(self):
        self.lm_feat_proj.weight.data.normal_(0, 0.001)
        self.item_emb.weight.data.normal_(0, 0.001)
        self.feat_emb.weight.data.normal_(0, 0.001)
        torch.clamp(self.lm_feat_proj.weight.data, min=-0.01, max=0.01)
        torch.clamp(self.feat_emb.weight.data, min=-0.01, max=0.01)
        torch.clamp(self.item_emb.weight.data, min=-0.01, max=0.01)

        for key in self.all_layers:
            try:
                for layer in self.all_layers[key]:
                    # Xavier Initialization for weights
                    layer.weight.data.normal_(0, 0.001)
                    layer.bias.data.normal_(0.0, 0.001)
                    torch.clamp(layer.bias.data, min=-0.01, max=0.01)
            except:
                pass
