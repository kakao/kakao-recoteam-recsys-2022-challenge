import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

def build_dataset(sessions, purchases=None):
    print('build dataset. dataset info: ', sessions.shape)
    item_feats = pd.read_csv('../parsed_data/item_features.csv', dtype=str)
    item_feats['feat_id'] = item_feats['feature_category_id'] + ':' + item_feats['feature_value_id']
    d = item_feats.groupby("item_id")['feat_id'].apply(list).reset_index()
    item_to_feats = {x:y for (x, y) in zip(d.item_id, d.feat_id)}

    def items_to_feats(item_ids, item_to_feats):
        ret = []
        for x in item_ids:
            ret.extend(item_to_feats[x[1]])
        return ret

    sessions['kp'] = [(x,y) for (x,y) in zip(sessions.date, sessions.item_id)]
    t = sessions.groupby('session_id').kp.apply(lambda x: [y for y in sorted(x)]).reset_index()
    q = sessions.groupby('session_id')['date'].apply(lambda x: list(sorted(x))[0])
    v = sessions.groupby('session_id')['date'].apply(lambda x: list(sorted(x))[-1])
    q = pd.merge(q, v, on='session_id')
    print(t.shape, q.shape,v.shape)
    ret = pd.merge(pd.DataFrame(t), pd.DataFrame(q), on='session_id')
    if purchases is not None:
        dataset = pd.merge(ret, purchases, on='session_id')
        dataset.columns = ['session_id', 'item_ids', 'date_session_begin', 'date_session_end', 'target_item', 'date_purchase']
        dataset['target_feat'] = dataset.target_item.apply(lambda x: item_to_feats[x])
    else:
        dataset = ret
        dataset.columns = ['session_id', 'item_ids', 'date_session_begin', 'date_session_end']

    dataset['feats'] = dataset.item_ids.apply(lambda x: items_to_feats(x, item_to_feats))

    dataset['ym'] = dataset.date_session_end.apply(lambda x:x[:7])
    dataset['year'] = dataset.date_session_end.apply(lambda x:x[:4])
    dataset['month'] = dataset.date_session_end.apply(lambda x:x[5:7])
    dataset['dow'] =  pd.to_datetime(dataset.date_session_end).dt.dayofweek.apply(lambda x: str(x))
    dataset.sort_values(by='date_session_end', inplace=True)
    return dataset

def convert_dataset(fname):
    df = joblib.load(fname)
    df["item_ids"] = df[["view_dates", "views"]].apply(lambda x: list(zip(x['view_dates'], x['views'])), axis=1)

    df["date_session_begin"] = df["view_dates"].apply(lambda x: x[0])

    df["target_item"] = df["purchase"]

    item_feats = pd.read_csv('../data/item_features.csv', dtype=str)
    item_feats['feat_id'] = item_feats['feature_category_id'] + ':' + item_feats['feature_value_id']
    d = item_feats.groupby("item_id")['feat_id'].apply(list).reset_index()
    item_to_feats = {x:y for (x, y) in zip(d.item_id, d.feat_id)}

    def items_to_feats(item_ids, item_to_feats):
        ret = []
        for x in item_ids:
            ret.extend(item_to_feats[x[1]])
        return ret

    df['feats'] = df.item_ids.apply(lambda x: items_to_feats(x, item_to_feats))
    

    df['target_feat'] = df.target_item.apply(lambda x: item_to_feats[x])

    df['ym'] = df.date_session_end.apply(lambda x:x[:7])
    df['year'] = df.date_session_end.apply(lambda x:x[:4])
    df['month'] = df.date_session_end.apply(lambda x:x[5:7])
    df['dow'] =  pd.to_datetime(df.date_session_end).dt.dayofweek.apply(lambda x: str(x))
    df.sort_values(by='date_session_end', inplace=True)
    
    
    df = df[["session_id", "item_ids", "date_session_begin", "date_session_end", "target_item", "date_purchase", "target_feat", "feats", "ym", "year", "month", "dow"]]
    return df

#tr_csv = convert_dataset("../processed/df_train_all")
#val_csv = convert_dataset("../processed/df_val")
#te_csv = convert_dataset("../processed/df_te")

tr_sessions = pd.read_csv('../parsed_data/train_sessions.csv', dtype=str)
tr_purchases = pd.read_csv('../parsed_data/train_purchases.csv', dtype=str)
tr_csv = build_dataset(tr_sessions, tr_purchases)

val_sessions = pd.read_csv('../parsed_data/val_sessions.csv', dtype=str)
val_purchases = pd.read_csv('../parsed_data/val_purchases.csv', dtype=str)
val_csv = build_dataset(val_sessions, val_purchases)

te_sessions = pd.read_csv('../parsed_data/te_sessions.csv', dtype=str)
te_purchases = pd.read_csv('../parsed_data/te_purchases.csv', dtype=str)
te_csv = build_dataset(te_sessions, te_purchases)

submit_tr_sessions = pd.read_csv('../parsed_data/train_sessions.csv', dtype=str)
submit_tr_purchases = pd.read_csv('../parsed_data/train_purchases.csv', dtype=str)
submit_tr_csv = build_dataset(submit_tr_sessions, submit_tr_purchases)

leaderboard_tr_sessions = pd.read_csv('../parsed_data/te_sessions.csv', dtype=str)
submit_leaderboard_csv = build_dataset(leaderboard_tr_sessions)

final_tr_sessions = pd.read_csv('../parsed_data/te_sessions.csv', dtype=str)
submit_final_csv = build_dataset(final_tr_sessions)

os.makedirs('./data', exist_ok=True)
os.makedirs('./data/submit', exist_ok=True)

joblib.dump(tr_csv, './data/tr_csv')
joblib.dump(val_csv, './data/val_csv')
joblib.dump(te_csv, './data/te_csv')

joblib.dump(submit_tr_csv, './data/submit_tr_csv')
joblib.dump(submit_leaderboard_csv, './data/submit_leaderboard_csv')
joblib.dump(submit_final_csv, './data/submit_final_csv')


item_feats = pd.read_csv('../parsed_data/item_features.csv', dtype=str)
items = sorted(item_feats.item_id.unique().tolist(), key=lambda x: int(x))
iid2idx = dict(zip(items, range(len(items))))
joblib.dump([iid2idx], 'submit-indices')

item_feats = pd.read_csv('../parsed_data/item_features.csv', dtype=str)
items = sorted(item_feats.item_id.unique().tolist(), key=lambda x: int(x))
iid2idx = dict(zip(items, range(len(items))))
joblib.dump([iid2idx], 'val-indices')
