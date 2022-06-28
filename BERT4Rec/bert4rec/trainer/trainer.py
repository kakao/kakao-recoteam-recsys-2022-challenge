import importlib
import os
import glob
import json
from typing import List, Optional, Tuple
from os.path import join as pjoin
import math

import numpy as np
import joblib
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert4rec.utils import get_logger
from bert4rec.model import BERT4Rec, BERTAdam
from bert4rec.data import (
    BERTMLMTrainDataset,
    BERTMLMEvalDataset,
    BERTNIPTrainDataset,
    BERTNIPEvalDataset,
    RecsysValidation,
    RecsysSubmitDataset
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # To suppress tensorflow warning.


class BERT4RecTrainer:
    def __init__(self,
                 use_amp: bool = True,
                 use_bert_adam: bool = False,
                 lr: float = 1e-4,
                 step_size: int = 10,
                 gamma: float = 1.0,
                 num_epochs: int = 100,
                 batch_size: int = 1024,
                 num_workers: int = 8,
                 ndcg_k: int = 30,
                 model_name: str = "",
                 log_dir: str = "",
                 custom_dataset: str= "",
                 current_timestamp: str="",
                 is_submit: bool = True
                 ):
        assert torch.cuda.is_available(), "CUDA should be available"
        self._use_amp = use_amp
        self._use_bert_adam = use_bert_adam
        self._lr = lr
        self._step_size = step_size
        self._gamma = gamma
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._ndcg_k = ndcg_k
        self._logger = get_logger()
        self._set_log_writer(log_dir, model_name)
        self._model_name = model_name
        self._keys = None
        self._model_opt = None
        self._custom_dataset = custom_dataset
        self._current_timestamp = current_timestamp
        self._is_submit = is_submit
        self._set_dataset()
    
    def _set_dataset(self):
        if self._custom_dataset:
            custom_dataset = importlib.import_module(f'bert4rec.data.custom_dataset.{self._custom_dataset}')
            self.MLMTrainDataset = custom_dataset.BERTMLMTrainDataset
            self.MLMEvalDataset = custom_dataset.BERTMLMEvalDataset
            self.NIPTrainDataset = custom_dataset.BERTNIPTrainDataset
            self.NIPEvalDataset = custom_dataset.BERTNIPEvalDataset
        else:
            self.MLMTrainDataset = BERTMLMTrainDataset
            self.MLMEvalDataset = RecsysValidation # BERTMLMEvalDataset
            self.NIPTrainDataset = BERTNIPTrainDataset
            self.NIPEvalDataset = RecsysValidation # BERTNIPEvalDataset
    
    def _set_candidiates(self):
        candidate_mask = torch.zeros(len(self._idmap), dtype=torch.bool)
        
        import pandas as pd
        target_cands = set(pd.read_csv("../data/candidate_items.csv", dtype=str).item_id.tolist())
        for cand_i in target_cands:
            if cand_i not in self._idmap:
                continue

            candidate_mask[self._idmap[cand_i]] = True

        self._candidate_mask = candidate_mask.cuda()

    def _set_log_writer(self, parent_log_dir, model_name):
        if not parent_log_dir:
            self._log_dir = None
        else:
            model_log_dir = pjoin(parent_log_dir, model_name)
            os.makedirs(model_log_dir, exist_ok=True)
            fnames = glob.glob(f"{model_log_dir}/*")
            num_prev = sum(1 for fname in fnames if os.path.isdir(fname))
            self._log_dir = pjoin(model_log_dir, f"version{num_prev}")
            self._smry_writer = SummaryWriter(self._log_dir)

    def _mlm_training_step(self, seqs, lbls, mask, last_view, *item_metas):
        logits = self._model(seqs, last_view, *item_metas)
        logits = logits.view(-1, logits.size(-1))
        lbls = lbls.view(-1)

        # mask = torch.zeros((logits.shape[0], logits.shape[1] + 2)).cuda()
        # mask = mask.scatter(1, seqs, 1)[:, :-2].bool()
        # mask = mask.view(-1, mask.size(-1))
        # logits[mask] = -10000

        loss = F.cross_entropy(logits, lbls, ignore_index=self._pad_token)
        return loss

    def _nip_training_step(self, seqs, lbls, mask, last_view, *item_metas):
        logits = self._model(seqs, last_view, *item_metas)
        # mask = torch.zeros((logits.shape[0], logits.shape[1] + 2)).cuda()
        # mask = mask.scatter(1, seqs, 1)[:, :-2].bool()
        # logits[mask] = 0
        loss = F.cross_entropy(logits, lbls.squeeze())
        return loss

    def _train_one_epoch(self) -> float:
        def calculate_batch_loss(batch):
            seqs, lbls, mask, last_view, *item_metas = map(lambda x: x.cuda(), batch)
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._training_step(seqs, lbls, mask, last_view, *item_metas)
            else:
                loss = self._training_step(seqs, lbls, mask, last_view, *item_metas)
            return loss, seqs.size(0)

        self._model.train()
        loss, total_size = 0.0, 0
        for batch in self._train_loader:
            batch_loss, batch_size = calculate_batch_loss(batch)
            self._optimizer.zero_grad()
            if self._use_amp:
                self._scaler.scale(batch_loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                batch_loss.backward()
                self._optimizer.step()
            loss += (batch_loss.item() * batch_size)
            total_size += batch_size
        self._lr_scheduler.step()
        loss /= total_size
        return loss

    @torch.no_grad()
    def _calculate_ndcg(self) -> float:
        def calculate_batch_ndcg(batch):
            seqs, answer, mask, last_view, *item_metas = map(lambda x: x.cuda(), batch)
            logits = self._model(seqs, last_view, *item_metas).squeeze(1)
            _, indices = torch.topk(logits, self._ndcg_k, 1)
            hits = (indices == answer).nonzero()[:, -1]
            return (1 / torch.log2(2 + hits)).sum().item(), seqs.size(0)

        self._model.eval()
        ndcg, total_size = 0.0, 0
        for batch in self._vali_loader:
            batch_ndcg, batch_size = calculate_batch_ndcg(batch)
            ndcg += batch_ndcg
            total_size += batch_size
        ndcg /= total_size
        return ndcg

    @torch.no_grad()
    def _calculate_mrr(self) -> float:
        def _calculate_batch_mrr(batch):
            seqs, answer, mask, last_view, *item_metas = map(lambda x: x.cuda(), batch)
            logits = self._model(seqs, last_view, *item_metas).squeeze(1)

            # mask = torch.zeros((logits.shape[0], logits.shape[1] + 2)).cuda()
            # mask = mask.scatter(1, seqs, 1)[:, :-2].bool()
            logits[mask] = -1000000
            logits[:, self._candidate_mask] += 10000
            
            _, indices = torch.topk(logits, self._ndcg_k, 1)
            hits = (indices == answer)
            div = torch.arange(1, 1 + self._ndcg_k).unsqueeze(0).cuda()
            mrr = (hits / div).sum(-1)
            return mrr.detach().cpu().numpy().mean(), seqs.size(0)

        self._model.eval()
        mrr, total_size = [], 0
        for batch in self._vali_loader:
            batch_mrr, batch_size = _calculate_batch_mrr(batch)
            mrr.append(batch_mrr)
            total_size += batch_size
        
        return np.mean(mrr)
    
    @torch.no_grad()
    def _generate_mask(self, data_loader, csv_path, name):

        def _generate_logit(batch):
            seqs, mask, last_view, *item_metas = map(lambda x: x.cuda(), batch)
            logits = self._model(seqs, last_view, *item_metas).squeeze(1)

            return logits.cpu().numpy(), mask.cpu().numpy()
        
        self._model.eval()

        mrr = self._calculate_mrr()
        self._logger.info(f"MRR@{self._ndcg_k} before save leaderboard: {mrr}")

        outputs = []
        masks = []
        for batch in data_loader:
            logits, mask = _generate_logit(batch)
            outputs.extend(logits)
            masks.append(mask)
        
        outputs = np.vstack(outputs)
        masks = np.vstack(masks)
        joblib.dump((outputs, masks), "save_logits_and_mask")
    
    @torch.no_grad()
    def _make_leaderboard(self, data_loader, csv_path, name):
        if name in ('val', 'test'):
            tony_idmap = joblib.load('val-indices')[0]
        else:
            tony_idmap = joblib.load('submit-indices')[0]

        mapper = [
            tony_idmap[item_id]
            for item_id, _ in sorted(self._idmap.items(), key=lambda x: x[1])
        ]

        def _calculate_batch_mrr(batch):
            if name in ('val', 'test'):
                seqs, answer, mask, last_view, *item_metas = map(lambda x: x.cuda(), batch)
            else:
                seqs, mask, last_view, *item_metas = map(lambda x: x.cuda(), batch)
            logits = self._model(seqs, last_view, *item_metas).squeeze(1)
            origin_logits = logits.clone().detach().cpu().numpy()
            row = np.full((origin_logits.shape[0], len(tony_idmap)), -100000, dtype=origin_logits.dtype)
            row[:, mapper] = origin_logits

            logits[mask] = -1000000
            logits[:, self._candidate_mask] += 10000
            
            _, indices = torch.topk(logits, self._ndcg_k, 1)
            indices = indices.detach().cpu().numpy()
            indices = np.vectorize(self._idmap_inv.get)(indices)
            return indices, row
        
        self._model.eval()

        mrr = self._calculate_mrr()
        self._logger.info(f"MRR@{self._ndcg_k} before save leaderboard: {mrr}")

        outputs = []
        origin_logits = []
        for batch in data_loader:
            _indices, _origin_logit = _calculate_batch_mrr(batch)
            outputs.extend(_indices)
            origin_logits.append(_origin_logit)
        
        origin_logits = np.vstack(origin_logits)

        leaderboard_csv = joblib.load(csv_path)
        
        save_dir = pjoin(f"../logits", name)
        os.makedirs(save_dir, exist_ok=True)

        logit_path = pjoin(save_dir, f"bert4rec_{self._current_timestamp}.logits")
        joblib.dump((list(leaderboard_csv.session_id), origin_logits), logit_path)

        return True
        

    def _train(self):

        mrr = self._calculate_mrr()
        self._logger.info(f"MRR@{self._ndcg_k} before training: {mrr}")

        tbar = trange(1, self._num_epochs + 1)
        for epoch in tbar:
            loss = self._train_one_epoch()
            # ndcg = self._calculate_ndcg()
            ndcg = self._calculate_mrr()
            metrics = {"train_loss": loss, f"NDCG@{self._ndcg_k}": ndcg}
            if math.isnan(loss):
                raise ValueError("loss is nan")
            tbar.set_postfix(metrics)
            if self._log_dir is not None:
                self._smry_writer.add_scalar("Loss/train", loss, epoch)
                self._smry_writer.add_scalar(f"NDCG@{self._ndcg_k}/Vali", ndcg, epoch)

    def _prepare(self,
                 data_dir: str,
                 mask_prob: float,
                 num_blocks: int,
                 dim: int,
                 num_heads: int,
                 drop_rate: float,
                 sequential: bool,
                 sequence_length: int,
                 is_mlm: bool,
                 use_timestamp: bool,
                 pretrained_model_file_fname: Optional[str],
                 pretrained_model_opt_fname: Optional[str],
                 timestamp_mode: str,
                 meta_item_info: Optional[List[Tuple[int, int, int]]],
                 session_meta_item_info: Optional[List[Tuple[int, int, int]]]
                 ):
        self._training_step = self._mlm_training_step if is_mlm else self._nip_training_step
        raw_dir = pjoin(os.path.dirname(data_dir), "raw")
        with open(pjoin(data_dir, "keys"), "r") as fin:
            self._keys = fin.read().splitlines()
            self._idmap = {key: idx for idx, key in enumerate(self._keys)}
            self._idmap_inv = {idx: key for idx, key in enumerate(self._keys)}
        self._set_candidiates()
        num_items = len(self._keys)
        cloze_token = num_items
        self._pad_token = num_items + 1
        if pretrained_model_opt_fname is not None:
            with open(pretrained_model_opt_fname, "r") as fin:
                model_opt = json.load(fin)
        else:
            model_opt = None

        if meta_item_info is not None:
            for idx, row in enumerate(meta_item_info):
                if row[-1] and row[0] is None:
                    if not os.path.isabs(row[-1]):
                        row[-1] = pjoin(data_dir, f"{row[-1]}.npy")

                    if model_opt:
                        row[1] = model_opt['meta_item_info'][idx][1]
                        row[2] = model_opt['meta_item_info'][idx][2]
                    else:
                        weight = np.load(row[-1])
                        row[1] = weight.shape[0]
                        row[2] = weight.shape[1]

        if session_meta_item_info is not None:
            for idx, row in enumerate(session_meta_item_info):
                if row[-1] and row[0] is None:
                    if not os.path.isabs(row[-1]):
                        row[-1] = pjoin(data_dir, f"{row[-1]}.npy")

                    if model_opt:
                        row[1] = model_opt['session_meta_item_info'][idx][1]
                        row[2] = model_opt['session_meta_item_info'][idx][2]
                    else:
                        weight = np.load(row[-1])
                        row[1] = weight.shape[0]
                        row[2] = weight.shape[1]
        self._logger.info(model_opt)
        self._logger.info(meta_item_info)
        self._logger.info(session_meta_item_info)
        self._model = BERT4Rec(
            num_items,
            num_blocks,
            dim,
            num_heads,
            drop_rate,
            sequential,
            sequence_length,
            is_mlm,
            meta_item_info,
            session_meta_item_info
        ).cuda()
        if pretrained_model_file_fname is not None:
            state_dict = torch.load(pretrained_model_file_fname)
            self._model.load_state_dict(state_dict, strict=False)
            
        val_csv_fname = "submit_val_csv_main" if self._is_submit else "val_csv_main"
        if is_mlm:
            train_dset = self.MLMTrainDataset(
                fname=pjoin(data_dir, "train"),
                fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
                cloze_token=cloze_token,
                pad_token=self._pad_token,
                sequence_length=sequence_length,
                mask_prob=mask_prob,
                timestamp_mode=timestamp_mode
            )
            vali_dset = self.MLMEvalDataset(
                fname=pjoin(raw_dir, val_csv_fname),
                fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
                cloze_token=cloze_token,
                sequence_length=sequence_length,
                timestamp_mode=timestamp_mode,
                pad_token=self._pad_token,
                keys_fname=pjoin(data_dir, "keys"),
            )
        else:
            train_dset = self.NIPTrainDataset(
                fname=pjoin(data_dir, "train"),
                fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
                cloze_token=cloze_token,
                sequence_length=sequence_length,
                timestamp_mode=timestamp_mode,
                pad_token=self._pad_token
            )
            vali_dset = self.NIPEvalDataset(
                fname=pjoin(raw_dir, val_csv_fname),
                fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
                cloze_token=cloze_token,
                sequence_length=sequence_length,
                timestamp_mode=timestamp_mode,
                pad_token=self._pad_token,
                keys_fname=pjoin(data_dir, "keys"),
            )
        val_experiment_dset = RecsysValidation(
            fname=pjoin(raw_dir, "val_csv_main"),
            fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
            cloze_token=cloze_token,
            sequence_length=sequence_length,
            timestamp_mode=timestamp_mode,
            pad_token=self._pad_token,
            keys_fname=pjoin(data_dir, "keys"),
        )
        test_experiment_dset = RecsysValidation(
            fname=pjoin(raw_dir, "te_csv_main"),
            fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
            cloze_token=cloze_token,
            sequence_length=sequence_length,
            timestamp_mode=timestamp_mode,
            pad_token=self._pad_token,
            keys_fname=pjoin(data_dir, "keys"),
        )
        last10000_dset = RecsysValidation(
            fname=pjoin(raw_dir, "submit_tr_last10000_csv_main"),
            fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
            cloze_token=cloze_token,
            sequence_length=sequence_length,
            timestamp_mode=timestamp_mode,
            pad_token=self._pad_token,
            keys_fname=pjoin(data_dir, "keys"),
        )

        leaderboard_dset = RecsysSubmitDataset(
            fname=pjoin(raw_dir, "submit_leaderboard_csv_main"),
            fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
            cloze_token=cloze_token,
            sequence_length=sequence_length,
            timestamp_mode=timestamp_mode,
            pad_token=self._pad_token,
            keys_fname=pjoin(data_dir, "keys"),
        )

        final_dset = RecsysSubmitDataset(
            fname=pjoin(raw_dir, "submit_final_csv_main"),
            fname_timestamp=pjoin(data_dir, "train_timestamp") if use_timestamp else "",
            cloze_token=cloze_token,
            sequence_length=sequence_length,
            timestamp_mode=timestamp_mode,
            pad_token=self._pad_token,
            keys_fname=pjoin(data_dir, "keys"),
        )
        self._train_loader = DataLoader(train_dset,
                                        batch_size=self._batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=self._num_workers)
        self._vali_loader = DataLoader(vali_dset,
                                       batch_size=self._batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self._num_workers)
        self._test_loader = DataLoader(test_experiment_dset,
                                       batch_size=self._batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self._num_workers)
        self._last10000_loader = DataLoader(last10000_dset,
                                       batch_size=self._batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self._num_workers)
        self._leaderboard_loader = DataLoader(leaderboard_dset,
                                       batch_size=self._batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self._num_workers)

        self._final_loader = DataLoader(final_dset,
                                       batch_size=self._batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self._num_workers)

        self._val_experiment_loader = DataLoader(val_experiment_dset,
                                       batch_size=self._batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self._num_workers)
        if self._use_bert_adam:
            self._optimizer = BERTAdam(self._model.parameters(), lr=self._lr)
        else:
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._lr_scheduler = optim.lr_scheduler.StepLR(self._optimizer, self._step_size, self._gamma)
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
        # Write model_opt to load trained model.
        self._model_opt = {
            "num_items": num_items,
            "num_blocks": num_blocks,
            "dim": dim,
            "num_heads": num_heads,
            "drop_rate": drop_rate,
            "sequential": sequential,
            "sequence_length": sequence_length,
            "is_mlm": is_mlm,
            "timestamp_mode": timestamp_mode,
            "meta_item_info": meta_item_info,
            "session_meta_item_info": session_meta_item_info
        }

        info = f"""
        [TRAINING INFO]
            num_items: {num_items}
            num_blocks: {num_blocks}
            dim: {dim}
            num_heads: {num_heads}
            drop_rate: {drop_rate}
            sequential: {sequential}
            sequence_length: {sequence_length}
            MLM: {is_mlm}
            USE BERTAdam: {self._use_bert_adam}
            lr: {self._lr}
            step_size: {self._step_size}
            gamma: {self._gamma}
            USE AMP(Automatic Mixed Precision): {self._use_amp}
            num_epochs: {self._num_epochs}
            batch_size: {self._batch_size}
            ndcg_k: {self._ndcg_k}
            log_dir: {self._log_dir}
            use_timestamp: {use_timestamp}
            timestamp_mode: {timestamp_mode}
            meta_item_info: {meta_item_info}
            """
        self._logger.info(info)

    def fit(self,
            data_dir: str,
            mask_prob: float,
            num_blocks: int,
            dim: int,
            num_heads: int,
            drop_rate: float,
            sequential: bool,
            sequence_length: int,
            is_mlm: bool,
            use_timestamp: bool,
            pretrained_model_file_fname: str = "",
            pretrained_model_opt_fname: str= "",
            timestamp_mode: str="",
            meta_item_info: Optional[List[Tuple[int, int, int]]] = None,
            session_meta_item_info: Optional[List[Tuple[int, int, int]]] = None,
            ):
        # prepare data and model
        self._prepare(
            data_dir,
            mask_prob,
            num_blocks,
            dim,
            num_heads,
            drop_rate,
            sequential,
            sequence_length,
            is_mlm,
            use_timestamp,
            pretrained_model_file_fname,
            pretrained_model_opt_fname,
            timestamp_mode,
            meta_item_info,
            session_meta_item_info
        )
        self._train()

    def save(self, save_dir: str):
        # Save option to load models
        with open(pjoin(save_dir, "model_opt.json"), "w") as fout:
            json.dump(self._model_opt, fout)

        # Save item keys
        with open(pjoin(save_dir, "keys"), "w") as fout:
            fout.write("\n".join(self._keys))

        # Save model"s weights
        torch.save(self._model.state_dict(), pjoin(save_dir, "model.pt"))
