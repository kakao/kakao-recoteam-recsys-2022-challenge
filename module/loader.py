import torch
import numpy as np
import pandas as pd
from easydict import EasyDict
from scipy.sparse import vstack
from torch.utils.data import Dataset, DataLoader


class SessionDataset(Dataset):
    def __init__(self, df, idmap, fidmap, num_negatives=0, mlp_params=None):
        super(SessionDataset, self).__init__()
        self.idmap = idmap
        self.fidmap = fidmap

        # def __idmap(x):
        #     return self.idmap[x]
        # self.idmapper = np.vectorize(__idmap)

        # def __to_tensor(a_list_of_lists):
        #     return [torch.tensor(x) for x in a_list_of_lists]

        # self.to_tensor = __to_tensor
        self.n_users = df.shape[0]
        # self.indices = range(self.n_users)
        self.n_items = len(self.idmap)
        self.num_negatives = num_negatives

        df = df.reset_index()
        self.views = df.views

        # def __getlen(x):
        #     return len(x)

        self.seq_lens = df.views.apply(len).tolist()
        self.hist_lens = df.history.apply(len).tolist()
        self.session_id = df.session_id
        self.histories = df.history
        self.columns = df.columns
        if 'purchase' in self.columns:
            self.purchase = df.purchase
        self.mlp_params = mlp_params

    def __getitem__(self, i):
        # view = self.idmapper(self.views[i])
        view = [self.idmap[x] for x in self.views[i]]
        seq_len = self.seq_lens[i]
        hist_len = self.hist_lens[i]
        session = self.session_id[i]
        # history = self.idmapper(self.histories[i])
        history = [self.idmap[x] for x in self.histories[i]]
        ret = {'views': view, 'seq_lens': seq_len, 'sessions': session, 'hist_lens': hist_len, 'histories': history}
        if self.mlp_params is not None:
            ret.update({
                'mat': {k: v[i] for k, v in self.mlp_params[0].items()},
                'cat': {k: v[0][i] for k, v in self.mlp_params[1].items()},
                'scalar': self.mlp_params[2][i]
            })
        if 'purchase' in self.columns:
            ret['purchases'] = self.idmap[self.purchase[i]]
        return ret

    def __len__(self):
        return self.n_users

    def collate(self, batch):
        batch_size = len(batch)
        ret = {'batch_size': batch_size, 'extra': {}}
        df_batch = pd.DataFrame(batch)
        def __totensor(x):
            return torch.tensor(x)
        ret['views'] = df_batch.views.apply(__totensor).tolist()
        hist_lens = torch.as_tensor(df_batch.hist_lens.tolist(), dtype=torch.int32)
        histories = df_batch.histories.apply(__totensor).tolist()
        ret['extra'] = {
            'seq_lens': torch.as_tensor(df_batch.seq_lens.tolist(), dtype=torch.int32),
            'histories': (torch.repeat_interleave(torch.arange(batch_size), hist_lens), torch.cat(histories)),
            'sessions': df_batch.sessions.tolist()
        }
        if 'purchase' in self.columns:
            ret['purchases'] = torch.tensor(df_batch.purchases.tolist())
            if self.num_negatives:
                negatives = torch.randint(high=self.n_items, size=(batch_size, self.num_negatives))
                while (negatives == ret['purchases'].unsqueeze(-1)).sum().item():
                    negatives = torch.randint(high=self.n_items, size=(batch_size, self.num_negatives))
                ret['extra']['negatives'] = negatives

        if self.mlp_params is not None:
            ret['mlp'] = {'mat': {}, 'cat': {}, 'scalar': []}
            ret['mlp']['scalar'] = torch.FloatTensor(np.vstack(df_batch.scalar.tolist()))

            df_cat = pd.DataFrame(df_batch.cat.tolist())
            ret['mlp']['cat']['dow'] = torch.tensor(df_cat.dow.tolist())
            ret['mlp']['cat']['hour'] = torch.tensor(df_cat.hour.tolist())
            df_mat = pd.DataFrame(df_batch.mat.tolist())
            for k in df_mat.columns:
                if isinstance(df_mat[k][0], np.ndarray):
                    ret['mlp']['mat'][k] = torch.FloatTensor(np.vstack(df_mat[k].tolist()))
                else:
                    ret['mlp']['mat'][k] = torch.FloatTensor(vstack(df_mat[k].tolist()).toarray())
        # print("ret", ret)
        return EasyDict(ret)


class SessionDataLoader(DataLoader):
    def __init__(self, df, idmap, fidmap, num_negatives=0, mlp_params=None, **kwargs):
        # self.ds = SessionDataset(df, idmap, fidmap, num_negatives=num_negatives, mlp_params=mlp_params)
        super(SessionDataLoader, self).__init__(df, collate_fn=df.collate, **kwargs)



def test(batch_size=128, mlp_params=False, debug=False, **kwargs):
    # python module/loader.py 64 --shuffle=False --num_workers=8 --pin_memory=True --persistent_workers=True --prefetch_factor=16
    import time
    import joblib
    from tqdm import tqdm
    from os.path import dirname, abspath
    dir = abspath(dirname(dirname(__file__)))
    print(f'Load from {dir}/processed')
    idmap, fidmap = joblib.load(f'{dir}/processed/indices')[:2]
    mlp_params = joblib.load(f'{dir}/processed/mlp_val') if mlp_params else None
    df = joblib.load(f'{dir}/processed/df_val')
    ds = SessionDataset(df, idmap, fidmap, mlp_params=mlp_params)
    dl = SessionDataLoader(ds, idmap, fidmap, batch_size=batch_size, mlp_params=mlp_params, **kwargs)
    start = time.time()
    for batch in tqdm(dl, ncols=70):
        if debug:
            import ipdb; ipdb.set_trace()
        pass
    elapsed = time.time() - start
    print(f'One epoch takes {elapsed} [sec]')


if __name__ == '__main__':
    import fire
    fire.Fire(test)
