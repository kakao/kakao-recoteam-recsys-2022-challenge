import os
import struct
import logging
from collections import Counter
from os.path import join as pjoin

import ray
import numpy as np


class MapReducer:
    def __init__(self,
                 fname: str,
                 tmp_dir: str,
                 pretrained_key_fname: str,
                 frequency_min_count: int,
                 consume_min_count: int,
                 sequence_length: int,
                 max_samples_per_stream: int,
                 num_workers: int):
        self._fname = fname
        self._tmp_dir = tmp_dir
        self._pretrained_key_fname = pretrained_key_fname
        self._frequency_min_count = frequency_min_count
        self._consume_min_count = consume_min_count
        self._sequence_length = sequence_length
        self._max_samples_per_stream = max_samples_per_stream
        self._num_workers = num_workers
        self._idmap = {}
        self._chunk_fnames = []

    def _split_chunks(self):
        # First, build item counts.
        counts = Counter()
        with open(self._fname, "r") as fin:
            for line in fin:
                stream = line.strip().split()
                counts.update(stream)

        # Second, load pretrained_keys if exists.
        use_pretrained_keys = self._pretrained_key_fname != ""
        if use_pretrained_keys:
            with open(self._pretrained_key_fname, "r") as fin:
                self._idmap = {key: idx for idx, key in enumerate(fin.read().splitlines())}

        # Third, filter out items from stream.
        num_lines = 0
        with open(self._fname, "r") as fin, open(f"{self._fname}.tmp", "w") as fout:
            for line in fin:
                stream = line.strip().split()
                stream = [item for item in stream if counts[item] >= self._frequency_min_count]
                if use_pretrained_keys:
                    stream = [item for item in stream if item in self._idmap]
                size = len(stream)
                if size < self._consume_min_count + 1:
                    continue
                if self._max_samples_per_stream > 0:
                    offset = max(0, size - 1 - self._sequence_length - (self._max_samples_per_stream - 1))
                    stream = stream[offset:]
                if not use_pretrained_keys:
                    for item in stream:
                        if item not in self._idmap:
                            self._idmap[item] = len(self._idmap)
                fout.write(f"{' '.join(stream)}\n")
                num_lines += 1

        assert num_lines != 0, "No data in given file."

        # Forth, split files into several chunks.
        q, r = divmod(num_lines, self._num_workers)
        num_lines_per_worker = [q for _ in range(self._num_workers)]
        if r != 0:
            for _r in range(r):
                num_lines_per_worker[_r] += 1

        if q == 0:
            num_lines_per_worker = num_lines_per_worker[r:]
            self._num_workers = len(num_lines_per_worker)

        count, worker_id = 0, 0
        with open(f"{self._fname}.tmp", "r") as fin:
            for line in fin:
                if count == 0:
                    chunk_fname = pjoin(self._tmp_dir, f"chunk_{worker_id}")
                    fout = open(chunk_fname, "w")
                    self._chunk_fnames.append(chunk_fname)
                fout.write(line)
                count += 1
                if count == num_lines_per_worker[worker_id]:
                    count = 0
                    worker_id += 1
                    fout.close()
        os.remove(f"{self._fname}.tmp")

    @staticmethod
    @ray.remote
    def _process(idmap, chunk_fname, sequence_length):
        fout_train = open(f"{chunk_fname}_train", "wb")
        fout_vali = open(f"{chunk_fname}_vali", "wb")

        pad_token = len(idmap) + 1
        buf = np.empty((sequence_length,), dtype="int32")
        with open(chunk_fname, "r") as fin:
            for line in fin:
                stream = line.strip().split()
                stream = [idmap[item] for item in stream if item in idmap]
                size = len(stream)
                num_iter = max(0, size - 1 - sequence_length) + 1
                for i in range(num_iter):
                    beg = max(0, size - 1 - i - sequence_length)
                    end = size - 1 - i
                    left = sequence_length - (end - beg)
                    buf[:left] = pad_token
                    buf[left:] = stream[beg:end]
                    fout_train.write(buf.tobytes())
                    if i == 0:
                        fout_vali.write(buf[1:].tobytes())
                        fout_vali.write(struct.pack("i", stream[-1]))

        fout_train.close()
        fout_vali.close()
        os.remove(chunk_fname)

    def _process_by_chunks(self):
        ray.init(num_cpus=self._num_workers, logging_level=logging.ERROR)
        idmap = ray.put(self._idmap)
        ray.get(
            [MapReducer._process.remote(idmap,
                                        chunk_fname,
                                        self._sequence_length)
             for chunk_fname in self._chunk_fnames]
        )
        ray.shutdown()

    def _aggregate(self, suffix: str):
        fout = open(pjoin(self._tmp_dir, suffix), "wb")
        for chunk_fname in self._chunk_fnames:
            fname = f"{chunk_fname}_{suffix}"
            with open(fname, "rb") as fin:
                while True:
                    data = fin.read(8192)
                    if data == b"":
                        break
                    fout.write(data)
            os.remove(fname)
        fout.close()

    def _write_keys(self):
        keys = sorted(self._idmap, key=self._idmap.get)
        with open(pjoin(self._tmp_dir, "keys"), "w") as fout:
            fout.write("\n".join(keys))

    def _map(self):
        self._split_chunks()
        self._process_by_chunks()

    def _reduce(self):
        self._aggregate("train")
        self._aggregate("vali")
        self._write_keys()

    def run(self):
        self._map()
        self._reduce()
