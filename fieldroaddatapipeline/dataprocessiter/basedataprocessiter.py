r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""
import functools
import itertools
import os
import queue
import threading
import warnings


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union,TYPE_CHECKING

import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
from  .. import SequentialSampler
def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0
__all__ = [
    "_BaseDataProcessIter",
]

class _BaseDataProcessIter:
    def __init__(self,processer) -> None:
        self._index_sampler = processer._index_sampler
        self._num_workers = processer.num_workers
        self._timeout = processer.timeout
        self._sampler_iter = iter(self._index_sampler)
        self._iter_len = self.__len__()
        self._num_yielded = 0
    def __iter__(self) -> '_BaseDataProcessIter':
        return self
    def _reset(self, processer, first_iter=False):
        self._sampler_iter = iter(processer._index_sampler)
        self._num_yielded = 0
    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        data = self._next_data()
        self._num_yielded += 1
        if self._num_yielded > self._iter_len:
            warn_msg = ("Length of IterableDataset was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format( self._iter_len,self._num_yielded)
        return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)
