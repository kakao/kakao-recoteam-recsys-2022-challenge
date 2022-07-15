import os
import joblib
import itertools
from os import makedirs
from collections import Counter
from os.path import join as pjoin

import fire
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

from module.utils import fix_seed


def tocsv(df, dir_name, fname,):
    df.to_csv(pjoin(dir_name, fname), index=False)


def train_test_split(submit=False):
    dtype_dict=str
    all_sessions = pd.read_csv('data/train_sessions.csv', dtype=dtype_dict)
    all_purchases = pd.read_csv('data/train_purchases.csv', dtype=dtype_dict)
    all_features = pd.read_csv('data/item_features.csv', dtype=dtype_dict)
    all_item_ids = set(all_sessions.item_id.tolist()) | set(all_purchases.item_id.tolist())
    last_montth_sessions = all_purchases[all_purchases.date.apply(lambda x: x[:7] == '2021-05')]
    last_month_session_ids = last_montth_sessions.session_id.unique().tolist()
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    os.makedirs(dir_name, exist_ok=True)
    val_session_ids = shuffle(last_month_session_ids)[:1000]
    tr_sessions = all_sessions[all_sessions.session_id.isin(val_session_ids) == False]
    tr_purchases = all_purchases[all_purchases.session_id.isin(val_session_ids) == False]
    val_sessions = all_sessions[all_sessions.session_id.isin(val_session_ids)]
    val_purchases = all_purchases[all_purchases.session_id.isin(val_session_ids)]
    tocsv(tr_sessions, dir_name, 'train_sessions.csv')
    tocsv(tr_purchases, dir_name, 'train_purchases.csv')
    tocsv(val_sessions, dir_name, 'val_sessions.csv')
    tocsv(val_purchases, dir_name, 'val_purchases.csv')
    if not submit:
        tocsv(te_sessions, dir_name, 'te_sessions.csv')
        tocsv(te_purchases, dir_name, 'te_purchases.csv')
    return dir_name


def build_feature_and_indices(feat_fname, root='processed'):
    df_feat = pd.read_csv(feat_fname, dtype=str)
    features = (df_feat.feature_category_id + ':' + df_feat.feature_value_id).unique()
    features = sorted(features.tolist(), key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))
    fid2idx = dict(zip(features, range(len(features))))
    items = sorted(df_feat.item_id.unique().tolist(), key=lambda x: int(x))
    iid2idx = dict(zip(items, range(len(items))))
    item_cv = CountVectorizer(analyzer=lambda x: x, vocabulary=iid2idx)
    iid2idx_inv = {v: k for k, v in iid2idx.items()}
    fid2idx_inv = {v: k for k, v in fid2idx.items()}

    # Featurize
    feat_cv = CountVectorizer(analyzer=lambda x: x, vocabulary=fid2idx)
    df_feat['feat_id'] = df_feat.feature_category_id + ':' + df_feat.feature_value_id
    feat_df = df_feat.groupby('item_id').feat_id.apply(list).reset_index()
    item_to_feats = {x: y for (x, y) in zip(feat_df.item_id, feat_df.feat_id)}
    vectors = feat_cv.fit_transform(feat_df.feat_id).toarray()
    iidx2vec = dict(zip(map(lambda x: iid2idx[x], feat_df.item_id.tolist()), vectors))
    iid2vec = dict(zip(feat_df.item_id.tolist(), vectors))
    features = np.array([iidx2vec[x] for x in range(len(iidx2vec))]).astype(np.float32)
    joblib.dump((iid2idx, fid2idx, iid2idx_inv, fid2idx_inv), f'{root}/indices')
    joblib.dump(features, f'{root}/features')
    joblib.dump((iid2vec, iidx2vec, item_to_feats), f'{root}/item2vec')


def build_dataset(fname_session, fname_purchase=None, keep_last=3, keep_last_date='2021-02'):
    df_sess = pd.read_csv(fname_session, dtype=str)
    history = df_sess.groupby('session_id').item_id.apply(lambda x: list(set(x)))
    # Drop duplicated views (keep last #)
    df_sort = df_sess.sort_values(['session_id', 'date'], ascending=[True, True])
    df_sess = df_sort.groupby(['session_id', 'item_id']).tail(keep_last)
    # Filtering views (lastest months)
    print("keep_last_date", keep_last_date)
    df_sess = df_sess[df_sess.date.apply(lambda x: x[:len(keep_last_date)]) >= keep_last_date]
    df_sess['kp'] = [(x,y) for (x,y) in zip(df_sess.date, df_sess.item_id)]
    left = df_sess.groupby('session_id').kp.apply(lambda x: [y for y in sorted(x)]).reset_index()
    if fname_purchase:
        df_purchase = pd.read_csv(fname_purchase, dtype=str)
        df = pd.merge(left, df_purchase, on='session_id')
    else:
        df = left
    df['view_dates'], df['views'] = list(zip(*df.kp.apply(lambda x: list(zip(*x)))))
    df = df.drop('kp', axis=1)
    df['date_session_end'] = df.view_dates.apply(lambda x: x[-1])
    add_columns = ['item_id', 'date'] if fname_purchase else []
    df = df[['session_id', 'views', 'view_dates', 'date_session_end'] + add_columns]
    add_columns = ['purchase', 'date_purchase'] if fname_purchase else []
    df.columns = ['session_id', 'views', 'view_dates', 'date_session_end'] + add_columns
    df = pd.merge(df, history, on='session_id')
    df.columns = ['session_id', 'views', 'view_dates', 'date_session_end'] + add_columns + ['history']
    df = df.sort_values('date_session_end')
    return df


def build_edges_v1(df_train, idmap, weights=True, save_fname=None, dir_name='processed'):
    if save_fname is None:
        save_fname = f'{dir_name}/edge_index_v1.{int(weights)}'
    print(f"Build edge_index at {save_fname}")
    view_edge_index = []
    for _, row in df_train.views.iteritems():
        for _prev, _next in zip(row, row[1:]):
            view_edge_index.append((idmap[_prev], idmap[_next]))
    if not weights:
        ret = (torch.LongTensor(view_edge_index).T, None)
        joblib.dump(ret, save_fname)
        return ret
    r_edges = []
    weights = []
    for edge, cnt in Counter(view_edge_index).items():
            r_edges.append(edge)
            weights.append(cnt)
    edge_index, edge_weight = torch.LongTensor(r_edges).T, torch.FloatTensor(weights)
    print(f'Edge shape is {edge_index.shape}')
    ret = (edge_index, edge_weight)
    joblib.dump(ret, save_fname)
    return ret


def build_edges_v2(df_train, idmap, weights=False, dir_name='processed'):
    save_fname = f'{dir_name}/edge_index_v2.{int(weights)}'
    print(f"Build edge_index at {save_fname}")
    df_tmp = df_train.apply(lambda x: [(v, x.purchase) for v in x.views], axis=1)
    vp_edge_index = []
    for _, view_purchase in df_tmp.iteritems():
        for v, p in view_purchase:
            vp_edge_index.append((idmap[v], idmap[p]))
    if not weights:
        ret = (torch.LongTensor(vp_edge_index).T, None)
        joblib.dump(ret, save_fname)
        return ret
    r_edges = []
    weights = []
    for edge, cnt in Counter(vp_edge_index).items():
            r_edges.append(edge)
            weights.append(cnt)
    ret = (torch.LongTensor(r_edges).T, torch.FloatTensor(weights))
    joblib.dump(ret, save_fname)
    return ret


def data_augmentation(df_train_fname, dir_name):
    print('Data augmentation is used')
    df = joblib.load(df_train_fname)
    def aug(val):
        val = pd.DataFrame(val)
        val = val[val.views.apply(len) > 0]
        val['purchase'] = val.views.apply(lambda x: x[-1])
        new_views, new_histories, new_view_dates = [], [], []
        zzz = zip(val.views.tolist(), val.view_dates.tolist(), val.history.tolist(), val.purchase.tolist())
        for _views, _view_dates, _histories, purchase in zzz:
            nv_row = []
            nvd_row =[]
            for view, view_date in zip(_views, _view_dates):
                if view != purchase:
                    nv_row.append(view)
                    nvd_row.append(view_date)
            nh_row = [x for x in _histories if x != purchase]
            new_views.append(nv_row)
            new_view_dates.append(nvd_row)
            new_histories.append(nh_row)
        val['views'] = new_views
        val['view_dates'] = new_view_dates
        val['history'] = new_histories
        val = val[val.views.apply(len) > 0]
        return val
    df = aug(df)
    df_aug = pd.concat([joblib.load(df_train_fname), df], axis=0)
    df_aug = df_aug.sort_values(by='date_session_end').reset_index(drop=True)
    joblib.dump(df_aug, f'{dir_name}/df_train_aug')
    return df_aug


def build_mlp_inputs(df, idmap, fidmap, item_to_feats, EMBED_DIM=16, save_fname=None, pops=None):
    def flatten(x):
        ret = []
        for k in x:
            ret.extend(k)
        return ret

    if pops is None:
        v_pop_all = Counter(flatten(flatten(df.views.tolist())))
        p_pop_all = Counter(df.purchase.tolist())
        v_pop_3m = Counter(flatten(df[df.date_session_end.apply(lambda x: x[:7]) >= "2021-05"].views.tolist()))
        p_pop_3m = Counter(df[df.date_session_end.apply(lambda x: x[:7]) >= "2021-05"].purchase.tolist())
        pops = v_pop_all, p_pop_all, v_pop_3m, p_pop_3m
    else:
        v_pop_all, p_pop_all, v_pop_3m, p_pop_3m = pops


    v_pop_all = df.views.apply(lambda x: [v_pop_all[y] for y in x])
    p_pop_all = df.views.apply(lambda x: [p_pop_all[y] for y in x])
    v_pop_3m = df.views.apply(lambda x: [v_pop_3m[y] for y in x])
    p_pop_3m = df.views.apply(lambda x: [p_pop_3m[y] for y in x])
    s_len = df.views.apply(lambda x: len(x))

    scalars = pd.concat([
        np.log(1 + s_len),
        np.log(1 + v_pop_all.apply(lambda x: np.mean(x))),
        np.log(1 + p_pop_all.apply(lambda x: np.mean(x))),
        np.log(1 + v_pop_all.apply(lambda x: np.max(x))),
        np.log(1 + p_pop_all.apply(lambda x: np.max(x))),
        np.log(1 + v_pop_3m.apply(lambda x: np.mean(x))),
        np.log(1 + p_pop_3m.apply(lambda x: np.mean(x))),
        np.log(1 + v_pop_3m.apply(lambda x: np.max(x))),
        np.log(1 + p_pop_3m.apply(lambda x: np.max(x)))
    ], axis=1).to_numpy()
    scalars = scalars

    item_cv = CountVectorizer(analyzer=lambda x: x, vocabulary=idmap)
    feat_cv = CountVectorizer(analyzer=lambda x: x, vocabulary=fidmap)
    item_all = item_cv.transform(df.views.apply(lambda x: x))
    last_20 = item_cv.transform(df.views.apply(lambda x: x[-20:]))
    last_5 = item_cv.transform(df.views.apply(lambda x: x[-5:]))
    last_2 = item_cv.transform(df.views.apply(lambda x: x[-2:]))
    last_1 = item_cv.transform(df.views.apply(lambda x: x[-1:]))
    last_5_feats = df.views.apply(lambda x: x[-5:]).apply(lambda x: flatten([item_to_feats[y] for y in x]))
    last_3_feats = df.views.apply(lambda x: x[-3:]).apply(lambda x: flatten([item_to_feats[y] for y in x]))
    last_1_feats = df.views.apply(lambda x: x[-1:]).apply(lambda x: flatten([item_to_feats[y] for y in x]))

    # year_map = {y: x for (x, y) in enumerate(['2020', '2021'])}
    # month_map = {y: x for (x, y) in enumerate(["%02d" % x for x in range(1, 13)])}
    hour_map = {y: x for (x, y) in enumerate(["%d" % x for x in range(0, 24)])}
    dow_map = {y: x for (x, y) in enumerate(map(str, range(7)))}
    df['dow'] = pd.to_datetime(df.date_session_end).dt.dayofweek.apply(lambda x: str(x))
    df['hour'] = pd.to_datetime(df.date_session_end).dt.hour.apply(lambda x: str(x))

    l5_feats = normalize(feat_cv.transform(last_5_feats), norm='l1').toarray()
    l3_feats = normalize(feat_cv.transform(last_3_feats), norm='l1').toarray()
    l1_feats = normalize(feat_cv.transform(last_1_feats), norm='l1').toarray()

    cats = {
        "dow": (df.dow.apply(lambda x: dow_map[x]).to_numpy(), (7, EMBED_DIM)),
        "hour": (df.hour.apply(lambda x: hour_map[x]).to_numpy(), (25, EMBED_DIM))
    }
    mats = {
        "item_bow": item_all,
        "item_last_20": last_20,
        "item_last_5": last_5,
        "item_last_2": last_2,
        "item_last_1": last_1,
        'item_last_5_feats': l5_feats,
        "item_last_3_feats": l3_feats,
        "item_last_1_feats": l1_feats
    }
    if save_fname:
        joblib.dump((mats, cats, scalars), save_fname)
        print(f"Dump {save_fname}")
    return mats, cats, scalars, pops


def run(submit=False, seed=2022):
    fix_seed(seed)
    if submit:
        root = 'data'
    else:
        root = 'parsed_data'
    print('Split train/test')
    if submit ==True:
        dir_name = train_test_split(submit=submit)
    else:
        import subprocess
        import os
        os.makedirs("./processed", exist_ok=True)
        os.system("cp ./parsed_data/* ./processed")
        dir_name = 'processed'
    yms = sorted(pd.read_csv(f'{dir_name}/train_purchases.csv').date.apply(lambda x: x[:7]).unique().tolist())
    keep_last_date = yms[-3]  # train 3 months
    yms = sorted(pd.read_csv(f'{dir_name}/val_purchases.csv').date.apply(lambda x: x[:7]).unique().tolist())
    keep_last_date_val_te = yms[-1]  # val 1 months

    build_feature_and_indices(f'{root}/item_features.csv', root=dir_name)
    print("Build features and indices")
    print("all")
    df_train_all = build_dataset(f'{dir_name}/train_sessions.csv', f'{dir_name}/train_purchases.csv', keep_last_date='2018')
    print("tr")
    df_train = build_dataset(f'{dir_name}/train_sessions.csv', f'{dir_name}/train_purchases.csv', keep_last_date=keep_last_date)
    print("val")
    df_val = build_dataset(f'{dir_name}/val_sessions.csv', f'{dir_name}/val_purchases.csv', keep_last_date=keep_last_date_val_te)
    joblib.dump(df_train_all, f'{dir_name}/df_train_all')
    joblib.dump(df_train, f'{dir_name}/df_train')
    joblib.dump(df_val, f'{dir_name}/df_val')
    if not submit:
        print('te')
        df_te = build_dataset(f'{dir_name}/te_sessions.csv', f'{dir_name}/te_purchases.csv', keep_last_date=keep_last_date_val_te)
        joblib.dump(df_te, f'{dir_name}/df_te')
    else:
        df_leader = build_dataset(f'{root}/test_leaderboard_sessions.csv')
        df_final = build_dataset(f'{root}/test_final_sessions.csv')
        joblib.dump(df_leader, f'{dir_name}/df_leader')
        joblib.dump(df_final, f'{dir_name}/df_final')
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    build_edges_v1(df_train_all, idmap, save_fname=f'{dir_name}/all_edge_index_v1.1')
    build_edges_v1(df_train, idmap, dir_name=dir_name)
    build_edges_v2(df_train, idmap, dir_name=dir_name)
    data_augmentation(f'{dir_name}/df_train', dir_name=dir_name)
    print("Build mlp inputs")
    item_to_feats = joblib.load(f'{dir_name}/item2vec')[-1]
    extra = ['leader', 'final'] if submit else ['te']
    pops = None
    for kind in ['train', 'train_aug', 'val'] + extra:
        if kind == 'train':
            _, _, _, pops = build_mlp_inputs(joblib.load(f'{dir_name}/df_{kind}'),
            idmap, fidmap, item_to_feats,
            save_fname=f'{dir_name}/mlp_{kind}', pops=pops)
        else:
            build_mlp_inputs(joblib.load(f'{dir_name}/df_{kind}'),
            idmap, fidmap, item_to_feats,
            save_fname=f'{dir_name}/mlp_{kind}', pops=pops)


if __name__ == '__main__':
    import time
    start = time.time()
    fire.Fire(run)
    elapsed = time.time() - start
    print(f'Preprocessing takes {elapsed} [sec]')
    makedirs('save', exist_ok=True)
