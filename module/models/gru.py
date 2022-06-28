import os
import time
import random
import joblib
from os.path import abspath, dirname, join as pjoin

import fire
import torch
from torch import nn
import  torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GRU(Model):
    def __init__(self, idmap, dim=64, gru_dim=48, dropout=0.1, **kwargs):
        super(GRU, self).__init__()
        self.idmap = idmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        self.gru_dim = gru_dim
        self.item_emb = nn.Embedding(self.n_items + 1, dim, padding_idx=self.n_items)
        self.item_bias = nn.Parameter(torch.zeros(self.n_items).float(), requires_grad=True)
        self.gru = nn.GRU(input_size=dim, hidden_size=self.gru_dim, num_layers=2)
        self.drop = nn.Dropout(dropout)
        self.last_layers = nn.ModuleList([
            nn.Linear(self.gru_dim, dim * 2),
            nn.Linear(dim * 2, dim) 
        ])
        self.last_bns = nn.ModuleList([
            nn.BatchNorm1d(dim * 2), 
            nn.BatchNorm1d(dim),
        ])
        self.init_weights()

    def session_emb(self, batch):
        # Get session embedding
        seq = pad_sequence(batch.views, batch_first=True, padding_value=self.n_items).to(self.device)
        _, indices = batch.extra.seq_lens.sort(dim=0, descending=True)
        packed = pack_padded_sequence(self.target_emb(seq[indices]), lengths=batch.extra.seq_lens[indices], batch_first=True)
        hs, h = self.gru(packed)
        h = h.transpose(0, 1)
        _, inv_indices = torch.sort(indices, 0, descending=False)
        # hs = hs[inv_indices]
        s_emb = self.drop(h[inv_indices].squeeze(dim=1))
        s_emb = s_emb.mean(dim=1)
        for l, bn in zip(self.last_layers, self.last_bns):
            s_emb = torch.tanh(self.drop(l(s_emb)))
            s_emb = bn(s_emb)
        return s_emb
    
    def forward(self, batch, mask=True):
        s_emb = self.session_emb(batch)
        logit = F.linear(s_emb, F.normalize(self.drop(self.item_emb.weight[:-1, :])), self.item_bias)
        if mask:
            logit[batch.extra.histories] = -10000.0
        return logit
    
    def target_emb(self, iid):
        return self.item_emb(iid)

    def init_weights(self):
        for name, weight in self.named_parameters():
            if len(weight.shape) > 1 and 'weight' in name:
                nn.init.xavier_uniform_(weight)
            elif 'bias' in name:
                weight.data.normal_(0.0, 0.001)
                torch.clamp(weight.data, min=-0.001, max=0.001)


def run(
    epochs=30, batch_size=512, device='cuda', seed=2022,
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/gru.pt', submit=False, **kwargs
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
    model = GRU(idmap=idmap).to(device)
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    ds_tr = SessionDataset(df_train, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_tr = SessionDataLoader(
        ds_tr, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )

    ds_te = SessionDataset(df_te, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap)
    #dl_te = SessionDataLoader(df_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap)
    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
    optimizer = torch.optim.Adam(model.parameters(),  lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-6, verbose=1)
    model.fit(dl_tr, dl_te, optimizer, scheduler, epochs=epochs, save_fname=save_fname, seed=seed)


if __name__ == '__main__':
    fire.Fire(run)
