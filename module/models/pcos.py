import math
import os
import random
import joblib
from os.path import abspath, dirname, join as pjoin

import fire
import itertools
import numpy as np
import torch
from scipy.sparse import csr_matrix
from collections import Counter, defaultdict

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from loader import SessionDataset, SessionDataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def flatten(t):
    return [item for sublist in t for item in sublist]


class PCos():
    def __init__(self, dir_name, save_fname):
        self.dir_name = dir_name
        self.save_fname = save_fname

    def _cos_similarity_logit(self, cos_sim, popularity, batch):
        score = cos_sim[[views[-1].item() for views in batch.views]]
        score = score + 0.0001 * popularity
        score = np.log(1e-6 + score)
        return torch.FloatTensor(score).to('cuda')

    def _disim_similarity_logit(self, DiSim, batch):
        score = DiSim[[views[-1].item() for views in batch.views]]
        score = np.log(1e-6 + score)
        return torch.FloatTensor(score).to('cuda')

    def fit(self, logit=False, kind='val'):
        print("loading resources...")
        idmap, fidmap = joblib.load(f'{self.dir_name}/indices')[:2]
        df_train = joblib.load(f'{self.dir_name}/df_train')
        df_val = joblib.load(f'{self.dir_name}/df_{kind}')
        iid2vec, iidx2vec, item2feat = joblib.load(f'{self.dir_name}/item2vec')

        print("creating item2vec based on genre popularity...")
        # calculate the scores for each genre
        common_genres = {k: 1/math.log(v+1) for k, v in Counter(flatten(item2feat.values())).most_common()}
        new_iid2vec2 = {}
        for k in idmap.keys():
            new_iid2vec2[k] = [score if genre in item2feat[k] else 0 for genre, score in common_genres.items()]
        new_iid2vec = {k: iid2vec[k] for k in idmap.keys()}
        
        print("calculating cos_sim...")
        if logit:
            svec = [new_iid2vec2[v] for v in idmap.keys()]
        else:
            svec = [new_iid2vec[v] for v in idmap.keys()]
        svec = np.array(svec)
        svec = torch.FloatTensor(svec)
        svec = svec.cuda()
        cos_sim = svec @ svec.T
        cos_sim = cos_sim.detach().cpu().numpy()
        
        if 'submit' in self.dir_name:
            ymmap = {
                "03": 0.3,
                "04": 0.6,
                "05": 0.9,
            }
        else:
            ymmap = {
                "02": 0.3,
                "03": 0.6,
                "04": 0.9,
            }
        rr = defaultdict(lambda: 0)
        for i, (k, ym, p) in enumerate(zip(
            df_train.views.apply(lambda x: [idmap[y] for y in x]).tolist(),
            df_train.date_session_end.apply(lambda x: x[5:7]).tolist(),
            df_train.purchase.apply(lambda x: idmap[x]).tolist())
                                      ):
            for j in set(k):
                rr[(i, j)] += ymmap[ym]
            rr[(i, p)] += ymmap[ym] * 3

        r = [x[0] for x in rr.keys()]
        c = [x[1] for x in rr.keys()]
        v = [x for x in rr.values()]
        view_matrix = csr_matrix((v, (r, c)), shape=(df_train.shape[0], len(item2feat)))
        Iij = np.dot(view_matrix.T, view_matrix)
        Iij = Iij.toarray()
        Ii = view_matrix.toarray().sum(0)
        confidence_array = (1+Iij) / (1 + np.expand_dims(Ii, 1))
        cos_sim = cos_sim * confidence_array
        np.fill_diagonal(cos_sim, 0)

        if logit:
            print("loading additional resources...")
            ds_val = SessionDataset(df_val, idmap=idmap, fidmap=fidmap, num_negatives=100)
            dl_te = SessionDataLoader(ds_val, batch_size=512, num_negatives=100, idmap=idmap, fidmap=fidmap)

            print("calculating item popularity...")
            mapper = np.vectorize(lambda x: idmap[x])
            res = [mapper(np.array(view)).tolist() for view in df_train[-65000:].views.tolist()]
            res2 = mapper(df_train.purchase.tolist())
            res = list(itertools.chain.from_iterable(res)) + res2.tolist()
            popularity = np.zeros(len(idmap))
            res = Counter(res)
            for i, cnt in res.items():
                popularity[i] = np.log(1 + cnt)

            print("calculating logits per session...")
            logit = []
            sessions = []
            for batch in dl_te:
                _logit = self._cos_similarity_logit(cos_sim, popularity, batch)
                torch.log(1e-10 + _logit)
                _logit[batch.extra.histories] = -10000.0
                logit.append(_logit.detach().cpu())
                sessions.extend(batch.extra.sessions)
            logit = torch.cat(logit, dim=0)
            logit = logit.numpy().astype(np.float32)

            joblib.dump((sessions, logit), self.save_fname)
            print(f"pcos logit (#session x #item) dumped at {self.save_fname}")

        else:
            joblib.dump(cos_sim, self.save_fname)
            print(f"pcos (#item x #item) dumped at {self.save_fname}")


def run(logit=False, submit=False, save_fname='save/DiSim', kind='val', **kwargs):
    print(locals())
    sleep_seconds = random.randint(1, 10)
    print(f'sleep {sleep_seconds} before start')
    # time.sleep(sleep_seconds)
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    dir_name = pjoin(abspath(dirname(dirname(dirname(__file__)))), dir_name)
    model = PCos(dir_name, save_fname)
    model.fit(logit, kind=kind)


if __name__ == '__main__':
    fire.Fire(run)
