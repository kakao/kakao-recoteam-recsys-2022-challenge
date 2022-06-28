import sys
import time
import random
import joblib
import itertools
from os import makedirs
from collections import Counter
from os.path import abspath, dirname, join as pjoin

import torch
import numpy as np
from tqdm import tqdm


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_fallback_items(df_train, df_val):
    tr = set(itertools.chain.from_iterable(df_train.views.tolist())).union(set(df_train.purchase.tolist()))
    val = set(itertools.chain.from_iterable(df_val.views.tolist()))
    if 'purchase' in df_val.columns:
        val = val.union(set(df_val.purchase.tolist()))
    fallback_items = val - tr
    if len(fallback_items) == 0:
        return None
    return fallback_items


def get_fallback_sessions_mask(batch, ratio=0.7, idmap_inv=None, fallback_items=None):
    mask = torch.zeros(len(batch.views), dtype=torch.bool)
    max_fallback_ratio = 0
    fallback_sessions = []
    mapper_inv = np.vectorize(lambda x: idmap_inv[x])
    fallback_sessions_indices = []
    for i, views in enumerate(batch.views):
        views = mapper_inv(views.numpy()).tolist()
        cnt = Counter(views)
        intersections = set(cnt.keys()).intersection(fallback_items)
        if not intersections:
            continue
        fallback_ratio = sum(cnt[v] for v in intersections) / float(len(views))
        max_fallback_ratio = max(max_fallback_ratio, fallback_ratio)
        if fallback_ratio > ratio:
            fallback_sessions_indices.append(i)
    if fallback_sessions_indices:
        fallback_sessions.extend(np.array(batch.extra.sessions)[fallback_sessions_indices].tolist())
    mask[fallback_sessions_indices] = True
    return mask


def extract_fallback_sessions(df_train, df_val, save_fname=None, ratio=0.7):
    tr = set(itertools.chain.from_iterable(df_train.views.tolist())).union(set(df_train.purchase.tolist()))
    val = set(itertools.chain.from_iterable(df_val.views.tolist()))
    if 'purchase' in df_val.columns:
        val = val.union(set(df_val.purchase.tolist()))
    fallback_items = val - tr
    fallback_sessions = []
    max_fallback_ratio = 0
    for session_id, views in tqdm(df_val.set_index('session_id').views.iteritems(), ncols=70):
        cnt = Counter(views)
        fallback_ratio = sum(cnt[v] for v in set(cnt.keys()).intersection(fallback_items)) / float(len(views))
        max_fallback_ratio = max(fallback_ratio, max_fallback_ratio)
        if fallback_ratio > ratio:
            fallback_sessions.append(session_id)
    df_fallback = df_val[df_val.session_id.isin(fallback_sessions)]
    if save_fname:
        joblib.dump(df_fallback, save_fname)
    return df_fallback


def forward(models, batch, mask=True):
    n_items = len(models[0].idmap)
    batch_size = len(batch.views)
    logit = torch.zeros(batch_size, n_items).to(models[0].device)
    for self in models:
        logit += self.forward(batch, mask=False)
    if mask:
        logit[batch.extra.histories] = -10000.0
    return logit


def validate(models, val_dl, topk=100, mask=True):
    from metric import get_hit_and_mrr
    [self.eval() for self in models]
    HIT, MRR = [], []
    for batch in tqdm(val_dl, ncols=70):
        logit = forward(models, batch, mask=mask)
        pred = (-logit).argsort()[:, :topk]
        hit, mrr = get_hit_and_mrr(pred, batch.purchases.to(models[0].device), topk, mean=False)
        HIT.extend(hit.tolist())
        MRR.extend(mrr.tolist())
    return np.mean(HIT), np.mean(MRR)


def export_logits(models, val_dl, save_fname, mask=False):
    [self.eval() for self in models]
    logit = []
    sessions = []
    for batch in tqdm(val_dl, ncols=70):
        _logit = forward(models, batch, mask=mask)
        logit.append(_logit.detach().cpu())
        sessions.extend(batch.extra.sessions)
    logit = torch.cat(logit, dim=0) / float(len(models))
    logit = logit.numpy().astype(np.float32)
    joblib.dump((sessions, logit), save_fname)


def result_from_models(save_fnames, all=False, valid=False, mask=True, submit=False, device='cuda', augmentation=False, kind='val', **valid_kwargs):
    sys.path.append(abspath(dirname(__file__)))
    from models.gru import GRU
    from models.gnn import GNN
    from models.mlp import MLP
    from models.grun import GRUN

    MODEL_MAP = {
        'grun': GRUN,
        'gru': GRU,
        'gnn': GNN,
        'mlp': MLP,
    }
    dir = pjoin(abspath(dirname(dirname(__file__))), 'processed')
    if submit:
        dir += '_submit'
    print(f'Load from {dir}')
    model_name = None
    if 'grun' in save_fnames[0]: 
        model_name = 'grun'
    elif 'gru' in save_fnames[0]:
        model_name = 'gru'
    elif 'gnn' in save_fnames[0]:
        model_name = 'gnn'
    elif 'mlp' in save_fnames[0]:
        model_name = 'mlp'
    model_kwargs = get_model_kwargs(model_name, all=all, dir=dir, augmentation=augmentation)
    models = []
    for save_fname in save_fnames:
        model = MODEL_MAP[model_name](**model_kwargs)
        splited = save_fname.split('.')
        save_fname_tmp = '.'.join(splited[:2 if not submit else 3] + ['tmp'] + [splited[-1]])
        try:
            success = model.load_state_dict(torch.load(save_fname))
        except:
            success = model.load_state_dict(torch.load(save_fname_tmp))
        print(save_fname, success)
        models.append(model.to(device))
    mname = type(models[0]).__name__
    dl_te = model.get_te_dataloader(512, dir_name=dir, kind=kind, mlp_params='mlp_params' in model_kwargs)
    if valid:
        hit, mrr = validate(models, dl_te, mask=mask, **valid_kwargs)
        print(f"[{mname}] HIT: {hit}, MRR: {mrr}")
        return hit, mrr
    else:
        makedirs(f'logits/{kind}', exist_ok=True)
        logit_fname = f'logits/{kind}/{mname.lower()}_{int(time.time())}.logits'
        export_logits(models, dl_te, logit_fname)
        print(f"Export logit {logit_fname} is succeeed")
        return logit_fname


def return_best_save_fnames(save_fnames, topn=10, **kwargs):
    fname_with_mrr = []
    for save_fname in save_fnames:
        hit, mrr = result_from_models([save_fname], valid=True, **kwargs)
        fname_with_mrr.append((mrr, save_fname))
    selected = sorted(fname_with_mrr, key = lambda x: -x[0])[:topn]
    mrrs, fnames = list(zip(*selected))
    print(f'MRRS: {mrrs}')
    return fnames


def get_model_kwargs(model_name, dir, all=False, augmentation=False):
    assert model_name in ['gru', 'gnn', 'grun', 'mlp']
    pick_edge = 'all_' if all else ''
    model_kwargs = {'idmap': joblib.load(f'{dir}/indices')[0]}
    if model_name in ['mlp']:
        model_kwargs['fidmap'] = joblib.load(f'{dir}/indices')[1]
        aug = '_aug' if augmentation else ''
        model_kwargs['mlp_params'] = joblib.load(f'{dir}/mlp_train{aug}')
    if model_name in ['grun', 'gnn']:
        model_kwargs['features'] = torch.FloatTensor(joblib.load(f'{dir}/features'))
        if model_name == 'grun':
            edge_index, edge_weight = joblib.load(f'{dir}/{pick_edge}edge_index_v1.1')
            model_kwargs['edge_index'] = edge_index
            model_kwargs['edge_weight'] = edge_weight
        elif model_name == 'gnn':
            edge_index = joblib.load(f'{dir}/{pick_edge}edge_index_v2.0')[0]
            model_kwargs['edge_index'] = edge_index
    print(f'keys: {model_kwargs.keys()}')
    return model_kwargs