import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert4rec.utils import get_logger
_logger = get_logger()

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 drop_rate: float):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, f"dim should be divided by num_heads. Got dim: {dim} / num_heads: {num_heads}"
        self._q = nn.Linear(dim, dim)
        self._k = nn.Linear(dim, dim)
        self._v = nn.Linear(dim, dim)
        self._fc = nn.Linear(dim, dim)
        self._drop = nn.Dropout(drop_rate)

        self._dim = dim
        self._num_heads = num_heads
        self._head_dim = dim // num_heads

    def forward(self,
                x: torch.FloatTensor,
                mask: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        # Input: N x S x D
        sequence_length = x.size(1)
        q, k, v = self._q(x), self._k(x), self._v(x)

        # N x H x S x (D // H)
        q = q.view(-1, sequence_length, self._num_heads, self._head_dim).transpose(1, 2)
        k = k.view(-1, sequence_length, self._num_heads, self._head_dim).transpose(1, 2)
        v = v.view(-1, sequence_length, self._num_heads, self._head_dim).transpose(1, 2)

        # N x H x S x S
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self._head_dim)
        if mask is not None:
            scores.masked_fill_(~mask, torch.finfo(torch.float16).min)
        scores = F.softmax(scores, dim=-1)
        scores = self._drop(scores)

        # N x S x H x (D // H)
        h = torch.matmul(scores, v).transpose(1, 2).contiguous()

        # N x S x D
        h = h.view(-1, sequence_length, self._dim)
        return self._fc(h)


class PositionWiseFeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 drop_rate: float):
        super(PositionWiseFeedForward, self).__init__()
        self._fc1 = nn.Linear(dim, 4 * dim)
        self._fc2 = nn.Linear(4 * dim, dim)
        self._drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # N x S x (4 x D)
        h = self._fc1(x)
        h = F.gelu(h)
        h = self._drop(h)
        # N x S x D
        return self._fc2(h)


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 drop_rate: float):
        super(TransformerBlock, self).__init__()
        self._attn = MultiHeadAttention(dim, num_heads, drop_rate)
        self._ln1 = nn.LayerNorm(dim)
        self._pffn = PositionWiseFeedForward(dim, drop_rate)
        self._ln2 = nn.LayerNorm(dim)
        self._drop = nn.Dropout(drop_rate)

    def forward(self,
                x: torch.FloatTensor,
                mask: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        # Sublayer with Attention
        h = self._attn(x, mask)
        h = self._drop(h)
        h = self._ln1(x + h)

        # Sublayer with PFFN
        h = self._pffn(h)
        return self._ln2(h + self._drop(h))


class BERT4Rec(nn.Module):
    def __init__(self,
                 num_items: int,
                 num_blocks: int,
                 dim: int,
                 num_heads: int,
                 drop_rate: float,
                 sequential: bool,
                 sequence_length: int,
                 is_mlm: bool,
                 meta_item_info: Optional[List[Tuple[int, int, int, str, bool]]],
                 session_meta_item_info: Optional[List[Tuple[int, int, int, str, bool]]],
                 ):
        super(BERT4Rec, self).__init__()
        self._pad_token = num_items + 1
        embed_size = num_items + 2  # +1 for cloze token, +1 for pad token

        if meta_item_info is not None:
            self._meta_item_embeds, self._meta_item_total_dim = self._init_meta(meta_item_info, is_mlm)
        else:
            self._meta_item_embeds = []
            self._meta_item_total_dim = 0

        if session_meta_item_info is not None:
            self._session_meta_item_embeds, self._session_meta_item_total_dim = self._init_meta(session_meta_item_info, is_mlm)
        else:
            self._session_meta_item_embeds = []
            self._session_meta_item_total_dim = 0
        
        self._item_embed = nn.Embedding(embed_size, dim, padding_idx=self._pad_token)
        self._total_item_dim = dim + self._meta_item_total_dim

        self._ln = nn.LayerNorm(self._total_item_dim)
        self._sequential = sequential
        self._sequence_length = sequence_length
        self._is_mlm = is_mlm
        if self._sequential:
            pos_embed = torch.rand(sequence_length, dim)
            nn.init.xavier_normal_(pos_embed)
            self._pos_embed = nn.Parameter(pos_embed)

        self._trms = nn.ModuleList([
            TransformerBlock(self._total_item_dim, num_heads, drop_rate)
            for _ in range(num_blocks)
        ])

        self._fc1 = nn.Linear(self._total_item_dim, dim)
        self._fc2 = nn.Linear(dim, num_items)
        self._fc2.weight = nn.Parameter(self._item_embed.weight[:-2])
        
    
    def _init_meta(self, meta_item_info, is_mlm):
        meta_embeds = []
        _meta_item_total_dim = 0 
        for _meta_info in meta_item_info:
            if len(_meta_info) >= 4:
                _meta_item_pad, _meta_item_cnt, _meta_item_dim, pretrain_weight_fname = _meta_info
            else:
                _meta_item_pad, _meta_item_cnt, _meta_item_dim = _meta_info
                pretrain_weight_fname = None

            if pretrain_weight_fname:
                if is_mlm:
                    weight = np.load(pretrain_weight_fname)
                    _meta_item_total_dim += weight.shape[-1]
                    weight = torch.from_numpy(weight)
                    _embed = nn.Embedding.from_pretrained(weight, freeze=True)
                else:
                    _meta_item_total_dim += _meta_item_dim
                    _embed = nn.Embedding(_meta_item_cnt, _meta_item_dim)
                    _embed.weight.requires_grad = False
            else:
                _embed = nn.Embedding(_meta_item_cnt, _meta_item_dim, padding_idx=_meta_item_pad)    
                _meta_item_total_dim += _meta_item_dim
            meta_embeds.append(_embed)

        return nn.ModuleList(meta_embeds), _meta_item_total_dim
    
    def _idx2embed(self, x: torch.FloatTensor, item_meta_list: Tuple[torch.FloatTensor]):
        h = self._item_embed(x)

        if self._sequential:
            h.add_(self._pos_embed)

        item_metas_embeds = [
            self._meta_item_embeds[idx](item_metas)
            for idx, item_metas in enumerate(item_meta_list)
            if idx < len(self._meta_item_embeds)
        ]
        if len(item_metas_embeds):
            return torch.concat([h] + item_metas_embeds, axis=2)
        else:
            return h
    
    def _session_embed(self, session_metas):
        session_metas_embeds = self._session_meta_item_embeds[0](session_metas)
        return session_metas_embeds

    def forward(self, x: torch.FloatTensor, session_metas, *item_metas) -> torch.FloatTensor:
        # pad token would be ignored in attention mechanism.
        # ~mask: pad token
        mask = x.ne(self._pad_token).unsqueeze(1).repeat(1, self._sequence_length, 1).unsqueeze(1)
        h = self._idx2embed(x, item_metas)

        h = self._ln(h)

        for trm in self._trms:
            h = trm(h, mask)

        if not(self.training and self._is_mlm):
            h = h[:, -1, :]

        h = self._fc2(F.gelu(self._fc1(h)))
        # h = F.gelu(self._fc1(h))
        # h = F.linear(h, F.normalize(self._fc2.weight))

        return h 
