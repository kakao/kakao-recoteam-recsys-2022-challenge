import sys
import time
import random
import joblib
import itertools
from os import makedirs
from collections import Counter
from os.path import abspath, dirname, exists, join as pjoin

import fire
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

import sys
sys.path.append(abspath(dirname(dirname(dirname(__file__)))))
from module.metric import get_hit_and_mrr
from module.loader import SessionDataset, SessionDataLoader
from module.utils import get_fallback_sessions_mask, get_fallback_items
from module.loader import SessionDataset, SessionDataLoader
from module.models.pcos import PCos


class EnsembleLogit:
    def __init__(self, logit_fnames, folder, device='cpu'):
        self.models = {}
        self.device = device
        self.idmap, self.fidmap, self.idmap_inv, self.feat_idmap_inv = joblib.load(f'{folder}/indices')
        self.mapper = np.vectorize(lambda x: self.idmap[x])
        for x in logit_fnames:
            if len(x) == 3:
                key, logit_fname, weight = x
                self.set_logic(key, logit_fname, weight=weight)
            else:
                key, logit_fname = x
                self.set_logic(key, logit_fname)
        self.folder = folder
        print(f'Load {logit_fnames}')
        self.set_popularity(folder)
        self.logit_min = 100.0
        self.logit_max = -100.0
        self.pcos_sim = None
        self.fallback_items = None
    
    def set_fallback(self, folder, kind='val'):
        self.fallback_items = get_fallback_items(joblib.load(f'{folder}/df_train'), joblib.load(f'{folder}/df_{kind}'))
                    
    def set_pcos(self, pcos_similarity):
        self.pcos_sim = pcos_similarity
        # feat = joblib.load(f'{self.folder}/features')
        # self.pcos_sim = feat @ feat.T
    
    def set_popularity(self, folder):
        df_train = joblib.load(f'{folder}/df_train')
        res = [self.mapper(np.array(view)).tolist() for view in df_train[-65000:].views.tolist()]
        res2 = self.mapper(df_train.purchase.tolist())
        res = list(itertools.chain.from_iterable(res)) + res2.tolist()
        popularity = np.zeros(len(self.idmap))
        res = Counter(res)
        for i, cnt in res.items():
            popularity[i] = np.log(1 + cnt)
        self.popularity = torch.FloatTensor(popularity)

    def similarity_logit(self, batch, mask=True, pool=None):
        score = 0.9 * torch.FloatTensor(self.pcos_sim[[views[-1].item() for views in batch.views]]).to(self.device)
        score += 0.09 * torch.FloatTensor(self.pcos_sim[[views[-min(len(views),2)].item() for views in batch.views]]).to(self.device)
        score += 0.01 * torch.FloatTensor(self.pcos_sim[[views[-min(len(views),3)].item() for views in batch.views]]).to(self.device)
        score = torch.log(1e-6 + score)
        score = score + 0.0001 * self.popularity.to(self.device)
        if mask:
            score[batch.extra.histories] = -10000.0
        if pool is not None:
            score[:, pool] += 10000.0
        return score
    
    def set_logic(self, key, logit_fname, weight=1.0):
        sessions, logit = joblib.load(logit_fname)
        self.models[key] = {'s2i': dict(zip(sessions, range(len(sessions))))}
        self.models[key].update({
            'logit': logit,
            'mapper': np.vectorize(lambda x: self.models[key].s2i[x]),
            'weight': float(weight)
        })
        self.models[key] = EasyDict(self.models[key])
        print(f'Set {key} logic')
        
    def get_logit(self, key, session_ids):
        model = self.models[key]
        return torch.FloatTensor(model.logit[model.mapper(session_ids)]).to(self.device)
    
    def get_weight(self, key):
        return self.models[key].weight

    def set_weights(self, weight_dict):
        for k in self.models.keys():
            self.models[k].weight = weight_dict.get(f'{k}', self.models[k].weight)
        weights = []
        for k in self.models.keys():
            weights.append((k, self.models[k].weight))
        print(f'Set {weights}')  
    
    def forward(self, batch, mask=True,  pool=None, fork_fallback=False, fallback_ratio=0.8):
        logits = [self.get_weight(key) * self.get_logit(key, batch.extra.sessions).unsqueeze(dim=-1) for key in self.models.keys() if self.get_weight(key) > 0.]
        if logits:
            logit = torch.cat(logits, dim=-1).mean(dim=-1)
        else:
            logit = self.similarity_logit(batch)
        if fork_fallback:
            fallback_mask = get_fallback_sessions_mask(batch, ratio=fallback_ratio, idmap_inv=self.idmap_inv, fallback_items=self.fallback_items)
            # overwrite logit for cold_sessions
            logit[fallback_mask] = self.similarity_logit(batch)[fallback_mask]
        self.logit_min = min(self.logit_min, logit.min().detach().cpu().item())
        self.logit_max = max(self.logit_max, logit.max().detach().cpu().item())
        if mask:
            logit[batch.extra.histories] = -100000.0
        if pool is not None:
            logit[:, pool] += 100000.0
        return logit
    
    def predict(self, batch, mask=True, pool=None, topk=100, fork_fallback=False):
        logit = self.forward(batch, mask=mask, pool=pool, fork_fallback=fork_fallback)
        pred = (-logit).argsort()[:, :topk]
        return pred
    
    def validate(self, val_dl, topk=100, mask=True, fork_fallback=False, fallback_ratio=0.8, only_warm=False, only_fallback=False, tqdm_off=False):
        HIT, MRR = [], []
        if tqdm_off:
            pbar = val_dl
        else:
            pbar = tqdm(val_dl, ncols=70)
        for batch in pbar:
            batch_size = len(batch.views)
            val_mask = torch.ones(batch_size, dtype=torch.bool)
            pred = self.predict(batch, mask=mask, topk=topk)
            fallback_mask = None
            if self.fallback_items is not None:
                fallback_mask = get_fallback_sessions_mask(batch, ratio=fallback_ratio, idmap_inv=self.idmap_inv, fallback_items=self.fallback_items)
            if self.fallback_items is not None and (fork_fallback or only_fallback):
                logit = self.similarity_logit(batch, mask=mask)
                pred[fallback_mask] = (-logit).argsort()[:, :topk][fallback_mask]
                if only_fallback:
                    val_mask = fallback_mask
                assert not only_warm
            if only_warm:
                if fallback_mask is not None:
                    val_mask = ~fallback_mask
            hit, mrr = get_hit_and_mrr(pred[val_mask], batch.purchases.to(self.device)[val_mask], topk, mean=False)
            HIT.extend(hit.tolist())
            MRR.extend(mrr.tolist())
        return np.mean(HIT) if HIT else 0., np.mean(MRR) if MRR else 0
    
    @classmethod
    def get_dataloader(cls, batch_size, folder, kind='val'):
        df_val = joblib.load(f'{folder}/df_{kind}')
        mlp_params = joblib.load(f'{folder}/mlp_{kind}')
        idmap, fidmap, _, _ = joblib.load(f'{folder}/indices')
        ds_te =  SessionDataset(df_val, idmap, fidmap, mlp_params=mlp_params)
        dl_te = SessionDataLoader(ds_te, idmap, fidmap, batch_size=batch_size, mlp_params=mlp_params)
        return dl_te
    
    def submit_to_csv(model, dl, topk=100, save_fname='submit.result', folder='data', **kwargs):
        import numpy as np
        import pandas as pd
        print(f'kwargs {kwargs}')
        mapper = np.vectorize(lambda x: model.idmap[x])
        candidates = pd.read_csv(f'{folder}/candidate_items.csv', dtype='str').item_id.unique()
        candidates = mapper(np.array(sorted(candidates, key=lambda x: int(x))))
        MSG = 'session_id,item_id,rank\n'
        for batch in tqdm(dl, ncols=70):
            pred = model.predict(batch, topk, pool=candidates, **kwargs)
            sess = batch.extra.sessions
            for i, p in enumerate(pred.detach().cpu().numpy().tolist()):
                for rank, j in enumerate(p, start=1):
                    msg = f'{sess[i]},{model.idmap_inv[j]},{rank}\n'
                    MSG += msg
        save_fname = f'{save_fname}-{int(time.time())}'
        with open(f'save/{save_fname}', 'w') as fout:
            fout.write(MSG)
        print(f"Dump 'save/{save_fname}'")
        df = pd.read_csv(f'save/{save_fname}', dtype='str')
        assert not bool(df.isna().sum().sum())
        pool = set(df.item_id.unique())
        cand = set(pd.read_csv(f'{folder}/candidate_items.csv', dtype='str').item_id.unique())
        assert len(pool - cand) == 0
        print(f'logit (min: {model.logit_min}, max: {model.logit_max})')


def main(kind='val', submit=False, **kwargs):
    from glob import glob
    from module.utils import result_from_models
    makedirs('tmp', exist_ok=True)
    makedirs(f'logits/{kind}', exist_ok=True)
    print(kwargs)
    logits = []
    for model_name in ['gru', 'grun', 'gnn', 'mlp', 'grun-all', 'mlp-augmentation']:
        _logit_fnames = glob(f'save/{model_name}.pt*')
        print(_logit_fnames)
        if _logit_fnames:
            model_kwargs = {}
            if len(model_name.split('-', maxsplit=1)) != 1:
                _, arguments = model_name.split('-', maxsplit=1)
                model_kwargs = {x: True for x in arguments.split('-')}
            print('model_kwargs:', model_kwargs)
            logit_fname = result_from_models(_logit_fnames, submit=submit, kind=kind, **model_kwargs)
            logits.append((model_name, logit_fname))
    print(joblib.dump(logits, f'tmp/logit_fnames-{int(time.time())}'))
    folder = 'processed'
    if submit:
        folder += '_submit'
    for save_fname in [f'logits/{kind}/pcos.logits', f'logits/{kind}/pcos.similarity']:
        if exists(save_fname): continue
        pcos = PCos(dir_name=folder, save_fname=save_fname)
        pcos.fit('similarity' not in save_fname, kind)
    logits.append(('pcos', f'logits/{kind}/pcos.logits'))
    bert_logits = glob(f'logits/{kind}/bert4rec_*.logits')
    if bert_logits:
        bert_logit = sorted(bert_logits, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        logits.append(('bert', bert_logit))
    model = EnsembleLogit(logits, folder=folder, device='cuda')
    model.set_fallback(folder=folder, kind=kind)
    if model.fallback_items is not None:
        print(f'Size of fallback items: {len(model.fallback_items)}')
    model.set_pcos(joblib.load(f'logits/{kind}/pcos.similarity'))
    dl = model.get_dataloader(512, folder, kind=kind)
    if kind == 'val':
        hit, mrr = model.validate(dl, **kwargs)
        print(f'HIT: {hit}, MRR: {mrr}')
    else:
        assert kind in ['leader', 'final']
        model.submit_to_csv(dl, fork_fallback=True)

if __name__ == '__main__':
    fire.Fire(main)
