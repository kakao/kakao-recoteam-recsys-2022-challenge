import os
import time
import fire
import joblib
import random
from os.path import abspath, dirname, join as pjoin

import torch
from torch import nn
from easydict import EasyDict
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader
from utils import get_fallback_items

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GNN(Model):
    def __init__(self, idmap, features, edge_index, hidden_channels=512, out_channels=256, dropout=0.1):
        super(GNN, self).__init__()            
        self.idmap = idmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        self.x = torch.nn.Parameter(features, requires_grad=False)
        self.edge_index = torch.nn.Parameter(edge_index, requires_grad=False)
        self.convs = nn.ModuleList([
            SAGEConv(-1, hidden_channels),
            SAGEConv(-1, out_channels)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels), 
            nn.BatchNorm1d(out_channels),
        ])
        self.drop = nn.Dropout(dropout)
        self.cross_entropy = nn.CrossEntropyLoss()

    def _forward(self):
        x = self.x
        for conv, bn in zip(self.convs, self.bns):
            # h = self.drop(conv(x, self.edge_index, self.edge_weight).tanh())
            h = self.drop(conv(x, self.edge_index).tanh())
            x = bn(h)
        return x
    
    def session_emb(self, batch, E=None):
        if E is None:
            E = self._forward()
        E = torch.vstack((E, E.mean(0)))
        views = pad_sequence(batch.views, batch_first=True, padding_value=self.n_items).to(self.device)
        Eviews = F.embedding(views, E, padding_idx=self.n_items)
        mask = views != self.n_items
        return (Eviews * mask.unsqueeze(-1)).sum(dim=1) / batch.extra.seq_lens.unsqueeze(dim=-1).to(self.device)
    
    def get_embeddings(self, batch):
        E = self._forward()
        ret = {'sessions': self.session_emb(batch, E), 'positives': F.embedding(batch.purchases.to(self.device), E)}
        if batch.extra.get('negatives', None) is not None:
            ret.update({'negatives': F.embedding(batch.extra.negatives.to(self.device), E)})
        return EasyDict(ret)

    def forward(self, batch, mask=True, normalize_sessions=False):
        E = self._forward()
        Esess = self.session_emb(batch, E)
        if normalize_sessions:
            Esess = F.normalize(Esess)
        logit = Esess @ F.normalize(E.T)
        # logit = Esess @ E.T
        if mask:
            logit[batch.extra.histories] = -10000.0
        return logit


def run(
    epochs=50, batch_size=512, device='cuda', seed=2022,
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/gnn.pt', submit=False, **kwargs
):
    print(locals())
    sleep_seconds = random.randint(1, 10)
    print(f'sleep {sleep_seconds} before start')
    time.sleep(sleep_seconds)
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    dir_name = pjoin(abspath(dirname(dirname(dirname(__file__)))), dir_name)
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}')
    df_te = joblib.load(f'{dir_name}/df_val')
    features = torch.FloatTensor(joblib.load(f'{dir_name}/features'))
    edge_index = joblib.load(f'{dir_name}/edge_index_v2.0')[0]
    model = GNN(idmap, features, edge_index).to(device)
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    ds_tr = SessionDataset(df_train, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_tr = SessionDataLoader(
        ds_tr, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    ds_te = SessionDataset(df_te, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap)
    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
    optimizer = torch.optim.Adam(model.parameters(),  lr=2e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-6, verbose=1)
    model.fit(dl_tr, dl_te, optimizer, scheduler, epochs=epochs, save_fname=save_fname, seed=seed)


if __name__ == '__main__':
    fire.Fire(run)
