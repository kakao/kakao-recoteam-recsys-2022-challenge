import os
import time
import itertools
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import torch.nn.functional as F

from metric import get_hit_and_mrr
from utils import get_fallback_items
from utils import get_fallback_sessions_mask
from torch import nn
import pytorch_lightning as pl



class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_items = None

    def predict(self, batch, topk=100, mask=True, pool=None):
        logit = self.forward(batch, mask=False)
        if mask:
            logit[batch.extra.histories] = -100000.0
        if pool is not None:
            logit[:, pool] += 100000.0
        pred = (-logit).argsort()[:, :topk]
        return pred

    def cross_entropy_loss(self, batch):
        batch_size = len(batch.purchases)
        logit = self.forward(batch)
        if batch.extra.get('negatives', None) is not None:
            num_negatives = batch.extra.negatives.shape[-1]
            rows = torch.repeat_interleave(torch.arange(batch_size), num_negatives)
            cols = batch.extra.negatives.flatten()
            logit[rows, cols] = -10000.0
        pos_loss = F.log_softmax(logit, dim=-1)[torch.arange(batch_size), batch.purchases].mean()
        loss = -pos_loss
        return loss

    def fit(self, tr_dl, te_dl, optimizer, scheduler, epochs, save_fname, topk=100, seed=2022):
        from tqdm import tqdm
        from utils import fix_seed
        fix_seed(seed)
        timestamp = int(time.time())
        print(f'start, save {save_fname}.{timestamp}')
        MRR = 0
        for epoch in range(1, epochs + 1):
            self.train()
            pbar = tqdm(tr_dl, ncols=70, leave=False)
            total_loss = []
            for batch in pbar:
                if tr_dl.pin_memory:
                    batch = EasyDict(batch)
                self.zero_grad()
                loss = self.cross_entropy_loss(batch)
                loss.backward()
                total_loss.append(loss.detach().cpu().item())
                optimizer.step()
            self.eval()
            hit, mrr = self.validate(te_dl, topk)

            if scheduler is not None:
                scheduler.step(mrr)

            total_loss = np.mean(total_loss)
            msg = f'Epoch {epoch}, HIT: {hit:.6f}, MRR: {mrr:.8f}, LOSS: {total_loss}'
            if mrr > MRR:
                torch.save(self.state_dict(), save_fname + f'.tmp.{timestamp}')
                MRR = mrr
                print(msg + ' (checkpoint)')
            elif epoch % 5 == 0:
                print(msg)
        if MRR:
            print(f'END, save {save_fname}.{timestamp}')
            os.rename(save_fname + f'.tmp.{timestamp}', save_fname + f'.{timestamp}')
        return self

    def similarity_logit(self, batch, mask=True, pool=None, popularity=True, confidence=False):
        _queries = [views[-1].item() for views in batch.views]
        score = self.similarity[_queries].to(self.device)
        if confidence:
            score = score * self.confidence[_queries].to(self.device)
        if popularity:
            score = score + 0.00002 * self.popularity.to(self.device)
            score = torch.log(1e-8 + score)
        if mask:
            score[batch.extra.histories] = -100000.0
        if pool is not None:
            score[:, pool] += 100000.0
        return score

    def set_popularity(self, df_train):
        res = [self.mapper(np.array(view)).tolist() for view in df_train[-65000:].views.tolist()]
        res2 = self.mapper(df_train.purchase.tolist())
        res = list(itertools.chain.from_iterable(res)) + res2.tolist()
        popularity = np.zeros(len(self.idmap))
        res = Counter(res)
        for i, cnt in res.items():
            popularity[i] = np.log(1 + cnt)
        self.popularity = torch.FloatTensor(popularity)
    
    def set_confidence(self, df_train, idmap):
        from scipy.sparse import csr_matrix
        r = []
        c = []
        v = []
        for i, k in enumerate(df_train.views.apply(lambda x: [idmap[y] for y in x]).tolist()):
            for j in set(k):
                r.append(i)
                c.append(j)
                v.append(1)
        view_matrix = csr_matrix((v, (r,c)), shape=(df_train.shape[0], len(idmap)))
        Iij = np.dot(view_matrix.T, view_matrix).toarray()
        Ii = view_matrix.toarray().sum(0)
        confidence_array = (1 + Iij) / (1 + np.expand_dims(Ii, 1))
        self.confidence = confidence_array

    def set_fork_logic(self, df_train, df_val, features):
        self.fallback_items = get_fallback_items(df_train, df_val)
        self.mapper = np.vectorize(lambda x: self.idmap[x])
        self.set_popularity(df_train)
        self.features = torch.FloatTensor(features)
        self.similarity = (self.features.to(self.device) @ self.features.T.to(self.device)).detach().cpu()

    def validate(self, val_dl, topk=100, mask=True, fork_fallback=False, fallback_ratio=0.8, only_warm=False, only_fallback=False, tqdm_off=False):
        self.eval()
        HIT, MRR = [], []
        if tqdm_off:
            pbar = val_dl
        else:
            pbar = tqdm(val_dl, ncols=70)
        for batch in pbar:
            batch_size = len(batch.views)
            val_mask = torch.ones(batch_size, dtype=torch.bool)
            pred = self.predict(batch, mask=mask, topk=topk)
            if self.fallback_items is not None:
                fallback_mask = get_fallback_sessions_mask(batch, ratio=fallback_ratio, idmap_inv=self.idmap_inv, fallback_items=self.fallback_items)
            if self.fallback_items is not None and (fork_fallback or only_fallback):
                logit = self.similarity_logit(batch, mask=mask)
                pred[fallback_mask] = (-logit).argsort()[:, :topk][fallback_mask]
                if only_fallback:
                    val_mask = fallback_mask
                assert not only_warm
            if only_warm:
                val_mask = ~fallback_mask
            hit, mrr = get_hit_and_mrr(pred[val_mask], batch.purchases.to(self.device)[val_mask], topk, mean=False)
            HIT.extend(hit.tolist())
            MRR.extend(mrr.tolist())
        self.train()
        return np.mean(HIT) if HIT else 0., np.mean(MRR) if MRR else 0

    def get_te_dataloader(self, batch_size, dir_name='processed', mlp_params=False, kind='val'):
        import joblib
        from loader import SessionDataLoader, SessionDataset
        assert kind in ['val', 'te', 'leader', 'final']
        df_te = joblib.load(f'{dir_name}/df_{kind}')
        mlp_params = joblib.load(f'{dir_name}/mlp_{kind}') if mlp_params else None
        idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
        ds_te = SessionDataset(df_te, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params)
        return SessionDataLoader(ds_te, batch_size=batch_size, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params)
