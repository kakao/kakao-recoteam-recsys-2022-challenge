import datetime
import os
import glob
import shutil
import importlib
from os.path import join as pjoin
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


from bert4rec.utils import get_logger

TIMESTAMP_PAD_TOKENS = {
    'WEEKDAY': 7,
    'DAY': 32,
    'HOUR': 24,
    'MINUTE': 60,
    'N_HOUR_BEFORE': 0
}

TIMESTAMP_META_CNT = {
    'WEEKDAY': 7,
    'DAY': 31,
    'HOUR': 24,
    'MINUTE': 60,
    'N_HOUR_BEFORE': None
}

class TimestampDataset:
    def __init__(self, fname_timestamp, sequence_length, timestamp_mode) -> None:
        self._data_timestamp = np.memmap(fname_timestamp, dtype="int32").reshape(-1, sequence_length)    
        if timestamp_mode == 'discretization':
            self.get = self._timestamp_discretization
        elif timestamp_mode == 'n_hour_before':
            self.get = self._timestamps_n_hour_before
        else:
            self.get = self._timestamp_n_hour_before_with_discretization

    def _timestamps_n_hour_before(self, index) -> List[int]:
        timestamps = self._data_timestamp[index]
        pivot_dt = datetime.datetime.fromtimestamp(max(timestamps))
        weekday = []
        n_hour_before = []
        for timestamp in timestamps:
            if timestamp == 0:
                weekday.append(TIMESTAMP_PAD_TOKENS['WEEKDAY'])
                n_hour_before.append(TIMESTAMP_PAD_TOKENS['N_HOUR_BEFORE'])
            else:
                dt = datetime.datetime.fromtimestamp(timestamp)
                weekday.append(dt.weekday())
                td = pivot_dt - dt
                td_hour = int(td.days * 24 + td.seconds//3600 + 1)
                n_hour_before.append(td_hour)
        return torch.LongTensor(weekday), torch.LongTensor(n_hour_before)

    def _timestamp_discretization(self, index) -> List[int]:
        timestamps = self._data_timestamp[index]
        weekday = []
        hour = []
        minute = []
        for timestamp in timestamps:
            if timestamp == 0:
                weekday.append(TIMESTAMP_PAD_TOKENS['WEEKDAY'])
                hour.append(TIMESTAMP_PAD_TOKENS['HOUR'])
                minute.append(TIMESTAMP_PAD_TOKENS['MINUTE'])
            else:
                dt = datetime.datetime.fromtimestamp(timestamp)
                weekday.append(dt.weekday())
                hour.append(dt.hour)
                minute.append(dt.minute)
        return torch.LongTensor(weekday), torch.LongTensor(hour), torch.LongTensor(minute)
    
    def _timestamp_n_hour_before_with_discretization(self, index) -> List[int]:
        timestamps = self._data_timestamp[index]
        pivot_dt = datetime.datetime.fromtimestamp(max(timestamps))
        weekday = []
        n_hour_before = []
        hour = []
        minute = []
        for timestamp in timestamps:
            if timestamp == 0:
                weekday.append(TIMESTAMP_PAD_TOKENS['WEEKDAY'])
                n_hour_before.append(TIMESTAMP_PAD_TOKENS['N_HOUR_BEFORE'])
                hour.append(TIMESTAMP_PAD_TOKENS['HOUR'])
                minute.append(TIMESTAMP_PAD_TOKENS['MINUTE'])
            else:
                dt = datetime.datetime.fromtimestamp(timestamp)
                weekday.append(dt.weekday())
                hour.append(dt.hour)
                minute.append(dt.minute)
                n_hour_before.append((pivot_dt - dt).seconds//3600 + 1)
        return torch.LongTensor(weekday), torch.LongTensor(n_hour_before), torch.LongTensor(hour), torch.LongTensor(minute)

# Dataset for MLM task.
class BERTMLMTrainDataset(Dataset):
    def __init__(self, fname: str, fname_timestamp:str, cloze_token: int, pad_token: int, sequence_length: int, mask_prob: float, timestamp_mode: str):
        self._data = np.memmap(fname, dtype="int32").reshape(-1, sequence_length)
        self.use_timestamp = len(fname_timestamp) != 0
        if (self.use_timestamp):
            self._data_timestamp = TimestampDataset(fname_timestamp, sequence_length, timestamp_mode)
        self._cloze_token = cloze_token
        self._pad_token = pad_token
        self._mask_prob = mask_prob

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, index):
        seqs, lbls, mask, last_view, = self._base(index)
        if self.use_timestamp:
            return seqs, lbls, mask, last_view, seqs, *self._data_timestamp.get(index)
        else:
            return seqs, lbls, mask, last_view, seqs
            

    def _base(self, index):
        seqs = torch.LongTensor(self._data[index])
        last_view = seqs[-2]
        lbls = torch.full_like(seqs, self._pad_token)

        probs = torch.rand_like(seqs, dtype=torch.float32)
        transform_mask = seqs.ne(self._pad_token)
        cloze_threshold = probs.less(self._mask_prob * 0.8)
        change_threshold = probs.less(self._mask_prob * 0.9)
        unchange_threshold = probs.less(self._mask_prob)

        cloze_indices = transform_mask & cloze_threshold
        change_indices = transform_mask & ~cloze_threshold & change_threshold
        unchange_indices = transform_mask & ~change_threshold & unchange_threshold

        lbls[cloze_indices] = seqs[cloze_indices]
        lbls[change_indices] = seqs[change_indices]
        lbls[unchange_indices] = seqs[unchange_indices]

        seqs[cloze_indices] = self._cloze_token
        seqs[change_indices] = torch.randint_like(seqs[change_indices], 0, self._cloze_token)
        mask = torch.zeros(self._pad_token+1, dtype=torch.bool).scatter_(0, seqs, True)[:-2]
        return seqs, lbls, mask, last_view


class BERTMLMEvalDataset(Dataset):
    def __init__(self, fname: str, fname_timestamp:str, cloze_token: int, sequence_length: int, timestamp_mode, pad_token):
        self._data = np.memmap(fname, dtype="int32").reshape(-1, sequence_length)
        self._cloze_token = torch.LongTensor([cloze_token])
        self._pad_token = pad_token

        self.use_timestamp = len(fname_timestamp) != 0
        if (self.use_timestamp):
            self._data_timestamp = TimestampDataset(fname_timestamp, sequence_length, timestamp_mode)

    def __len__(self) -> int:
        return self._data.shape[0]
    
    def __getitem__(self, index):
        query, answer, mask, last_view = self._base(index)
        if self.use_timestamp:
            return query, answer, mask, last_view, query, *self._data_timestamp.get(index)
        else:
            return query, answer, mask, last_view, query

    def _base(self, index):
        seqs = torch.LongTensor(self._data[index])
        last_view = seqs[-2]
        query, answer = seqs[:-1], seqs[-1:]
        query = torch.cat((query, self._cloze_token))
        mask = torch.zeros(self._pad_token+1, dtype=torch.bool).scatter_(0, query, True)[:-2]
        return query, answer, mask, last_view

class RecsysValidation(Dataset):
    def __init__(self, fname: str, fname_timestamp:str, cloze_token: int, sequence_length: int, timestamp_mode, pad_token, keys_fname):
        with open(keys_fname, "r") as fin:
            _keys = fin.read().splitlines()
        
        self._idmap = {key: idx for idx, key in enumerate(_keys)}
        streams = []
        with open(fname) as f:
            for line in f:
                stream = [self._idmap[item] for item in line.rstrip().split(' ') if item in self._idmap]

                stream = [pad_token] * (sequence_length - len(stream)) + stream
                stream = stream[-sequence_length:]
                streams.append(stream)
        
        self._data = np.array(streams, dtype="int32")
        self._cloze_token = torch.LongTensor([cloze_token])
        self._pad_token = pad_token

        self.use_timestamp = len(fname_timestamp) != 0
        if (self.use_timestamp):
            self._data_timestamp = TimestampDataset(fname_timestamp, sequence_length, timestamp_mode)

    def __len__(self) -> int:
        return self._data.shape[0]
    
    def __getitem__(self, index):
        query, answer, mask, last_view = self._base(index)
        if self.use_timestamp:
            return query, answer, mask, last_view, query, *self._data_timestamp.get(index)
        else:
            return query, answer, mask, last_view, query

    def _base(self, index):
        seqs = torch.LongTensor(self._data[index])
        last_view = seqs[-2]
        query, answer = seqs[:-1], seqs[-1:]
        query = torch.cat((query, self._cloze_token))
        mask = torch.zeros(self._pad_token+1, dtype=torch.bool).scatter_(0, query, True)[:-2]
        return query, answer, mask, last_view

    
class RecsysSubmitDataset(Dataset):
    def __init__(self, fname: str, fname_timestamp:str, cloze_token: int, sequence_length: int, timestamp_mode, pad_token, keys_fname):
        with open(keys_fname, "r") as fin:
            _keys = fin.read().splitlines()
        
        self._idmap = {key: idx for idx, key in enumerate(_keys)}
        streams = []
        with open(fname) as f:
            for line in f:
                stream = [self._idmap[item] for item in line.rstrip().split(' ') if item in self._idmap]

                stream = [pad_token] * (sequence_length - len(stream)) + stream
                stream = stream[-sequence_length:]
                streams.append(stream)
        
        self._data = np.array(streams, dtype="int32")
        self._cloze_token = torch.LongTensor([cloze_token])
        self._pad_token = pad_token
        self._sequence_length = sequence_length

        self.use_timestamp = len(fname_timestamp) != 0
        if (self.use_timestamp):
            self._data_timestamp = TimestampDataset(fname_timestamp, sequence_length, timestamp_mode)

    def __len__(self) -> int:
        return self._data.shape[0]
    
    def __getitem__(self, index):
        query, mask, last_view = self._base(index)
        if self.use_timestamp:
            return query, mask, last_view, query, *self._data_timestamp.get(index)
        else:
            return query, mask, last_view, query

    def _base(self, index):
        query = torch.LongTensor(self._data[index])
        last_view = query[-1]
        query = torch.cat((query, self._cloze_token))[-self._sequence_length:]
        mask = torch.zeros(self._pad_token+1, dtype=torch.bool).scatter_(0, query, True)[:-2]
        return query, mask, last_view

# Dataset for NIP(Next Item Prediction) task.
# These dataset class work same as MLM-Evaldataset.
# Just specify their name for clarity.
BERTNIPTrainDataset = BERTMLMEvalDataset
BERTNIPEvalDataset = BERTMLMEvalDataset


class DatasetBuilder:
    def __init__(self,
                 stream_data_fname: str,
                 stream_timestamp_data_fname: str,
                 save_dir: str,
                 frequency_min_count: int = 0,
                 consume_min_count: int = 0,
                 sequence_length: int = 0,
                 max_samples_per_stream: int = 0,
                 pretrained_key_fname: str = "",
                 mr_impl: str = "c",
                 num_workers: int = 8,
                 pretrained_features: Optional[Dict] = None,
                 ):
        """
        params:
            stream_data_fname: fname for stream data (e.g v2-mat-stream-1d's main)
            save_dir: directory to save preprocessed data.
            frequency_min_count: threshold to cut items with low frequency.
            consume_min_count: threshold to cut stream with few consume.
            sequence_length: max sequence length.
            max_samples_per_stream: max number of samples from each stream. 0 means no limit.
            pretrained_key_fname: fname for pretrained model's item keys.
            mr_impl: map-reduce implementation. ("c" for C++ / "ray" for ray)
            num_workers: number of workers to use in map-reduce.
        """
        self._stream_data_fname = stream_data_fname
        self._stream_timestamp_data_fname = stream_timestamp_data_fname
        self._save_dir = save_dir
        self._frequency_min_count = frequency_min_count
        self._consume_min_count = consume_min_count
        self._sequence_length = sequence_length
        self._max_samples_per_stream = max_samples_per_stream
        self._pretrained_key_fname = pretrained_key_fname
        self._mr_impl = mr_impl
        self._num_workers = num_workers
        self._pretrained_features = pretrained_features
        self._logger = get_logger()

        self._logger.info(f"stream_data_fname: {stream_data_fname}")
        self._logger.info(f"_stream_timestamp_data_fname: {stream_timestamp_data_fname}")
        self._logger.info(f"save_dir: {save_dir}")
        self._logger.info(f"frequency_min_count: {frequency_min_count}")
        self._logger.info(f"consume_min_count: {consume_min_count}")
        self._logger.info(f"sequence_length: {sequence_length}")
        self._logger.info(f"max_samples_per_stream: {max_samples_per_stream}")
        self._logger.info(f"map-reduce impl: {mr_impl}")
        self._logger.info(f"num_workers: {num_workers}")
    
    def _build_pretrained_feature(self):
        if not self._pretrained_features:
            return
        
        with open(pjoin(self._save_dir, 'keys'), "r") as fin:
            _keys = fin.read().splitlines()

        for pretrain_feature_info in self._pretrained_features:
            if pretrain_feature_info["custom_builder"]:
                package, module = pretrain_feature_info["custom_builder"].rsplit('.', 1)
                feature_builder = getattr(importlib.import_module(f'bert4rec.data.custom_dataset.{package}'), module)
            else:
                feature_builder = self._plain_pretrain_feature_builder
            feat = feature_builder(_keys, pretrain_feature_info)

            np.save(pjoin(self._save_dir, pretrain_feature_info["key"]), feat)
    
    def _plain_pretrain_feature_builder(self, keys, pretrain_feature_info):
        feat_fname = pjoin(pretrain_feature_info["dir_name"], "main")

        feat_list = []
        feats = feature.load(feat_fname)
        for _key in keys:
            feat_list.append(feats.get_feature(_key))
        
        return np.stack(feat_list)

    def build(self):
        self._logger.info("Processing stream data...")
        tmp_dir = pjoin(self._save_dir, "tmp")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        os.makedirs(self._save_dir, exist_ok=True)
        mr_module = importlib.import_module(f"bert4rec.data.{self._mr_impl}_mr")
        mr = mr_module.MapReducer(
            self._stream_data_fname,
            self._stream_timestamp_data_fname,
            tmp_dir,
            self._pretrained_key_fname,
            self._frequency_min_count,
            self._consume_min_count,
            self._sequence_length,
            self._max_samples_per_stream,
            self._num_workers
        )
        mr.run()
        for fname in glob.glob(pjoin(self._save_dir, "*")):
            if os.path.basename(fname) != "tmp":
                os.remove(fname)
        for fname in glob.glob(pjoin(tmp_dir, "*")):
            os.rename(fname, pjoin(self._save_dir, os.path.basename(fname)))
        shutil.rmtree(tmp_dir)
        self._build_pretrained_feature()
        self._logger.info("Processing stream data finished!")
