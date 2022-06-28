from os.path import join as pjoin

import numpy as np

import joblib

def feature_builder_item_gnn_feat(keys, pretrain_feature_info):
    feat_fname = pjoin(pretrain_feature_info["dir_name"], "item_gnn_feat")
    idmap, feat = joblib.load(feat_fname)

    
    feat_list = []
    mean_vector = feat.mean(axis=0)

    for key in keys:
        feat_list.append(feat[idmap[key]])

    feat_list.extend([mean_vector, mean_vector]) # for pad, clz token
    return np.stack(feat_list)