# cython: language_level = 3
# cython: boundscheck = False
# cython: nonecheck = False
# cython: wraparound = False
# distutils: language = c++

from typing import Union

from libcpp.string cimport string


cdef extern from "c_mr.hpp":
    cppclass CMapReducer:
        CMapReducer(string, string, string, string, int, int, int, int, int) except +
        void run() nogil except +


cdef class __MapReducerWrapper:
    cdef CMapReducer* obj

    def __cinit__(self,
                  string fname,
                  string fname_timestamp,
                  string tmp_dir,
                  string pretrained_key_fname,
                  int frequency_min_count,
                  int consume_min_count,
                  int sequence_length,
                  int max_samples_per_stream,
                  int num_workers):
        self.obj = new CMapReducer(fname, fname_timestamp, tmp_dir, pretrained_key_fname,
                                   frequency_min_count, consume_min_count, sequence_length,
                                   max_samples_per_stream, num_workers)

    def __dealloc__(self):
        del self.obj

    def run(self):
        self.obj.run()


class MapReducer:
    def __init__(self,
                 fname: Union[str, bytes],
                 fname_timestamp: Union[str, bytes],
                 tmp_dir: Union[str, bytes],
                 pretrained_key_fname: Union[str, bytes],
                 frequency_min_count: int,
                 consume_min_count: int,
                 sequence_length: int,
                 max_samples_per_stream: int = 0,
                 num_workers: int = 8,
                 ):
        if not isinstance(fname, bytes):
            fname = fname.encode("utf-8")
        if not isinstance(fname_timestamp, bytes):
            fname_timestamp = fname_timestamp.encode("utf-8")
        if not isinstance(tmp_dir, bytes):
            tmp_dir = tmp_dir.encode("utf-8")
        if not isinstance(pretrained_key_fname, bytes):
            pretrained_key_fname = pretrained_key_fname.encode("utf-8")

        self._obj = __MapReducerWrapper(fname, fname_timestamp, tmp_dir, pretrained_key_fname,
                                        frequency_min_count, consume_min_count, sequence_length,
                                        max_samples_per_stream, num_workers)

    def run(self):
        self._obj.run()
