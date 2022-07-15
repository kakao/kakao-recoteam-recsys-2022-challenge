import os
import time
import joblib
import numpy as np
import random
from os.path import abspath, dirname, join as pjoin

import fire
from numpy import ndarray
import torch
from torch import nn
import  torch.nn.functional as F

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader
from utils import get_fallback_items

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class MLP(Model):
    def __init__(self, idmap, fidmap, mlp_params, dim=256, layer_dim=3000, dropout=0.35):
        super(MLP, self).__init__()

        tr_matrices, tr_categoricals, tr_scalars = mlp_params

        self.idmap, self.fidmap = idmap, fidmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        self.n_items = len(self.idmap)
        self.item_emb = nn.Embedding(self.n_items, dim)
        self.item_emb.weight.data.normal_(0, 0.001)
        self.scalar_emb = nn.Linear(tr_scalars.shape[-1], 64)
        self.scalar_emb.weight.data.normal_(0, 0.001)
        self.scalar_bn = nn.BatchNorm1d(64)

        self.feat_emb = nn.Embedding(len(self.fidmap), dim)
        self.feat_emb.weight.data.normal_(0, 0.001)

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = 0
        aggr_width += 64

        for mat_id, mat in tr_matrices.items():
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(dim)])
            aggr_width += dim

        for cat_id, cat in tr_categoricals.items():
            _, (n_cats, embedding_dim) = cat
            cat_modules[cat_id] = nn.Embedding(n_cats, embedding_dim, padding_idx=-1)
            cat_bns[cat_id] = nn.BatchNorm1d(embedding_dim)
            aggr_width += embedding_dim

        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_layers = nn.ModuleDict(modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)
        self.last_layers = nn.ModuleList([
                nn.Linear(aggr_width, layer_dim),
                nn.Linear(layer_dim, dim)
            ])

        self.last_bns = nn.ModuleList([
                nn.BatchNorm1d(layer_dim),
                nn.BatchNorm1d(dim)
            ])
        self.item_bias = nn.Parameter(torch.zeros(self.n_items).float(), requires_grad=True)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def _forward(self, batch):
        # print(batch.mlp)
        for i, vs in batch.mlp.items():
            if isinstance(vs, dict):
                for j, v in vs.items():
                    # print("%s:%s" % (i, j), v.shape)
                    batch.mlp[i][j] = v.to(self.device)
            elif i == 'scalar':
                # print("scalar shape:", batch.mlp.scalar.shape)
                batch.mlp.scalar = batch.mlp.scalar.to(self.device)

        item_embs = F.normalize(self.item_emb.weight)
        feat_embs = self.feat_emb.weight

        aggr = []

        h = self.scalar_bn(self.scalar_emb(batch.mlp.scalar))
        aggr.append(h)

        h =  batch.mlp.mat['item_bow']
        h = self.all_bns['item_bow'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_5']
        h = self.all_bns['item_last_5'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_2']
        h = self.all_bns['item_last_2'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_1']
        h = self.all_bns['item_last_1'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_5_feats']
        h = self.all_bns['item_last_5_feats'][0](self.drop(F.linear(h, feat_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_3_feats']
        h = self.all_bns['item_last_3_feats'][0](self.drop(F.linear(h, feat_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_1_feats']
        h = self.all_bns['item_last_1_feats'][0](self.drop(F.linear(h, feat_embs.T)))
        aggr.append(h)

        for key, input in batch.mlp.mat.items():
            if key == 'item_bow' or key == 'item_last_5' or key == 'item_last_2' or key == 'item_last_1':
                continue

            if key == 'item_last_5_feats' or key == 'item_last_3_feats' or key == 'item_last_1_feats':
                continue
            if key == 'item_all_feats':
                continue

            h = input
            for l, bn in zip(self.all_layers[key], self.all_bns[key]):
                h = torch.tanh(self.drop(l(h)))
                h = bn(h)
            aggr.append(h)

        for key, input in batch.mlp.cat.items():
            h = self.drop(self.cat_embs[key](input))
            h = self.cat_bns[key](h)
            aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        for l, bn in zip(self.last_layers, self.last_bns):
            h = bn(torch.tanh(self.drop(l(h))))

        ret = F.linear(h, item_embs)
        return ret

    def init_weights(self):
        for key in self.all_layers:
            try:
                for layer in self.all_layers[key]:
                    # Xavier Initialization for weights
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.normal_(0.0, 0.001)
                    torch.clamp(layer.bias.data, min=-0.001, max=0.001)
            except:
                pass

    def forward(self, batch, mask=True):
        logit = self._forward(batch)
        if mask:
            logit[batch.extra.histories] = -10000.0
        return logit



def run(
    epochs=20, batch_size=256, device='cuda', seed=2022,
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/mlp.pt', submit=False, **kwargs
):
    print(locals())
    # sleep_seconds = random.randint(1, 10)
    # print(f'sleep {sleep_seconds} before start')
    # time.sleep(sleep_seconds)
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    dir_name = pjoin(abspath(dirname(dirname(dirname(__file__)))), dir_name)
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}')
    print(df_train.shape)
    df_te = joblib.load(f'{dir_name}/df_val')
    mlp_params = joblib.load(f'{dir_name}/mlp_train{aug}')
    model = MLP(idmap, fidmap, mlp_params=mlp_params, dim=256, layer_dim=3000, dropout=0.35).to(device)
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    print("train len:", len(df_train))
    ds = SessionDataset(df_train, idmap, fidmap, num_negatives=100, mlp_params=mlp_params)
    dl_tr = SessionDataLoader(
        ds, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    mlp_params_te = joblib.load(f'{dir_name}/mlp_val')
    ds_te = SessionDataset(df_te, idmap, fidmap, num_negatives=100, mlp_params=mlp_params_te)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params_te)

    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
    optimizer = torch.optim.Adam(model.parameters(),  lr=2 * 1e-4, weight_decay=5e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-6, verbose=1)
    model.fit(dl_tr, dl_te, optimizer, scheduler=None, epochs=epochs, seed=seed, save_fname=save_fname)


if __name__ == '__main__':
    fire.Fire(run)
