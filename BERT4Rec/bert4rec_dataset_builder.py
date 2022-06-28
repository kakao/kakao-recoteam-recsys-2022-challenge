import joblib
import pandas as pd
import os

import os
pretrain_path = 'train/data/recsys2022/pretrain-submit-b4r-plain-mlm-len20/raw'
os.makedirs(pretrain_path, exist_ok=True)
finetune_path = 'train/data/recsys2022/finetune-submit-b4r-plain-len20/raw'
os.makedirs(finetune_path, exist_ok=True)

val_pretrain_path = 'train/data/recsys2022/pretrain-b4r-plain-mlm-len20/raw'
os.makedirs(val_pretrain_path, exist_ok=True)
val_finetune_path = 'train/data/recsys2022/finetune-b4r-plain-len20/raw'
os.makedirs(val_finetune_path, exist_ok=True)


tr_csv = joblib.load('./data/tr_csv')
val_csv = joblib.load('./data/val_csv')
te_csv = joblib.load('./data/te_csv')

submit_tr_csv = joblib.load('./data/submit_tr_csv')
submit_leaderboard_csv = joblib.load('./data/submit_leaderboard_csv')
submit_final_csv = joblib.load('./data/submit_final_csv')

import datetime
def dt_str_2_ts_int(dt_strs):
    tss = []
    for dt_str in dt_strs:
        try:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            
        tss.append(str(int(datetime.datetime.timestamp(dt)*1000)))
    return tss

for csv_name, _csv in (
    ('tr_csv', tr_csv),
    ('tr_finetune_csv', tr_csv[-50000:]),
    ('val_csv', val_csv),
    ('te_csv', te_csv),
    ('submit_tr_csv', submit_tr_csv),
    ('submit_leaderboard_csv', submit_leaderboard_csv),
    ('submit_final_csv', submit_final_csv),
    ('submit_val_csv', submit_tr_csv[-2000:]),
    ('submit_tr_finetune_csv', submit_tr_csv[-50000:]),
    ('submit_tr_last10000_csv', submit_tr_csv[-10000:]),
):
    print('start convert', csv_name)
    if csv_name in ('submit_leaderboard_csv', 'submit_final_csv'):
        _csv['item_seq'] = _csv[['item_ids', ]].apply(lambda x: [y[1] for y in x['item_ids']], axis=1)
        _csv['ts_seq'] = _csv[['item_ids', ]].apply(lambda x: dt_str_2_ts_int([y[0] for y in x['item_ids']]), axis=1)
    else:
        _csv['item_seq'] = _csv[['item_ids', 'target_item']].apply(lambda x: [y[1] for y in x['item_ids']] + [x['target_item']], axis=1)
        _csv['ts_seq'] = _csv[['item_ids', 'date_purchase']].apply(lambda x: dt_str_2_ts_int([y[0] for y in x['item_ids']] + [x['date_purchase']]), axis=1)
    
    with open(f'{pretrain_path}/{csv_name}_main', 'w') as f:
        f.write( '\n'.join(_csv.item_seq.apply(lambda x: ' '.join(x))))

    with open(f'{pretrain_path}/{csv_name}_stream_timestamp', 'w') as f:
        f.write( '\n'.join(_csv.ts_seq.apply(lambda x: ' '.join(x))))
        
    with open(f'{val_pretrain_path}/{csv_name}_main', 'w') as f:
        f.write( '\n'.join(_csv.item_seq.apply(lambda x: ' '.join(x))))

    with open(f'{val_pretrain_path}/{csv_name}_stream_timestamp', 'w') as f:
        f.write( '\n'.join(_csv.ts_seq.apply(lambda x: ' '.join(x))))
        
    with open(f'{finetune_path}/{csv_name}_main', 'w') as f:
        f.write( '\n'.join(_csv.item_seq.apply(lambda x: ' '.join(x))))

    with open(f'{finetune_path}/{csv_name}_stream_timestamp', 'w') as f:
        f.write( '\n'.join(_csv.ts_seq.apply(lambda x: ' '.join(x))))

    with open(f'{val_finetune_path}/{csv_name}_main', 'w') as f:
        f.write( '\n'.join(_csv.item_seq.apply(lambda x: ' '.join(x))))

    with open(f'{val_finetune_path}/{csv_name}_stream_timestamp', 'w') as f:
        f.write( '\n'.join(_csv.ts_seq.apply(lambda x: ' '.join(x))))
