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
import glob
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union

import multiprocessing as python_multiprocessing
import torch
import json
import torch.multiprocessing as multiprocessing
from .sampler import (
    Sampler,
    SequentialSampler,)

from . import _utils
from .dataprocessiter import (
    _SingleProcessDataReaderIter,
    _MultiProcessingDataReaderIter,)
__all__ = [
    "FieldRoadDataReader",
]
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
from .dataset.basedataset import _DatasetFormat
def processjson(path):
    # 读取 JSON 文件
    with open(path['json'],'r') as file:
        coco = json.load(file)
    gnssnames=[os.path.join(path['gnss'], trajectory['file_name']) for trajectory in coco['trajectories']] if path['gnss'] is not None else None
    adjnames=[os.path.join(path['adj'], adj['file_name']) for adj in coco['adjs']] if path['adj'] is not None else None
    indices=[trajectory['index'] for trajectory in coco['trajectories']]
    filenames=dict(
        gnss = gnssnames,
        adj = adjnames,
        index = indices
    )
    return filenames

class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.

    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None

class FieldRoadDataReader:
    path: dict
    modet: str
    num_workers: int
    timeout: float
    sampler: Sampler
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataProcessIter']
    __initialized = False

    def __init__(self, path=None,dataset_format: str = None,mode: str = None,
                 max_len=None,drop_rate=None,scaler=None,num_workers: int = 0, 
                 timeout: float = 0, sampler=None,multiprocessing_context=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False):
        _all_dataset_format_ = ['folder','json',None]
        _all_mode_ = ['Trace','Point','Image',None]
        assert mode in _all_mode_
        assert dataset_format in _all_dataset_format_
        if dataset_format is None:
            dataset_format = 'json'
        if mode is None:
            mode = 'Trace'
        for i in range(len(_all_mode_)):
            if mode == _all_mode_[i]:
                mode = i 
        if mode != 0:
            if max_len !=None: 
                raise ValueError('max_len option should be None,because mode is not Trace; ')
            if drop_rate !=None:
                raise ValueError('drop_rate option should be None,because mode is not Trace; ')
        elif max_len == None :
            max_len = 5000
        elif drop_rate == None:
            drop_rate = 0.01
        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.')
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError('prefetch_factor option should be non-negative')

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.path = path
        self.dataset_format = dataset_format
        self.mode=mode
        self.max_len = max_len
        self.drop_rate = drop_rate
        self.scaler = scaler
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.multiprocessing_context = multiprocessing_context
        if isinstance(self.path, dict):
            if 'json' in self.path:
                self.filenames = processjson(self.path)
            else:
                self.filenames = dict(
                    gnss = sorted(glob.glob(os.path.join(self.path['gnss'], "*.xlsx"))) 
                    if self.path['gnss'] is not None else None,
                    adj = sorted(glob.glob(os.path.join(self.path['adj'], "*.npy"))) 
                    if self.path['adj'] is not None else None,
                    index = [None for i in range(len(glob.glob(os.path.join(self.path['gnss'], "*.xlsx"))))]
                )
        else:
            self.filenames = dict(
                gnss = sorted(glob.glob(os.path.join(self.path['gnss'], "*.xlsx"))) 
                if self.path['gnss'] is not None else None,
                adj = None,
                index = [None for i in range(len(glob.glob(os.path.join(self.path['gnss'], "*.xlsx"))))]
            )
        self.files_len = len(self.filenames['gnss'])
        if sampler is None:
            self.sampler = SequentialSampler(self.filenames['gnss'])
        else:
            self.sampler = sampler
        self.persistent_workers = persistent_workers
        self.__initialized = True
        self._iterator = None
        self.check_worker_number_rationality()
    def _get_iterator(self) -> '_BaseDataProcessIter':
        if self.num_workers == 0:
            return _SingleProcessDataReaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataReaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            'multiprocessing_context option '
                            f'should specify a valid start method in {valid_start_methods!r}, but got '
                            f'multiprocessing_context={multiprocessing_context!r}')
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError('multiprocessing_context option should be a valid context '
                                    'object or a string specifying the start method, but got '
                                    f'multiprocessing_context={multiprocessing_context}')
            else:
                raise ValueError('multiprocessing_context can only be used with '
                                 'multi-process loading (num_workers > 0), but got '
                                 f'num_workers={self.num_workers}')

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'path', 'mode', 'persistent_workers'):
            raise ValueError(f'{attr} attribute should not be set after {self.__class__.__name__} is initialized')
        super().__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataReaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid resetting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()
    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
         return self.sampler

    def __len__(self) -> int:
         return len(self._index_sampler)

    def check_worker_number_rationality(self):
        # This function check whether the dataloader's worker number is rational based on
        # current system's resource. Current rule is that if the number of workers this
        # Dataloader will create is bigger than the number of logical cpus that is allowed to
        # use, than we will pop up a warning to let user pay attention.
        #
        # eg. If current system has 2 physical CPUs with 16 cores each. And each core support 2
        #     threads, then the total logical cpus here is 2 * 16 * 2 = 64. Let's say current
        #     DataLoader process can use half of them which is 32, then the rational max number of
        #     worker that initiated from this process is 32.
        #     Now, let's say the created DataLoader has num_works = 40, which is bigger than 32.
        #     So the warning message is triggered to notify the user to lower the worker number if
        #     necessary.
        #
        #
        # [Note] Please note that this function repects `cpuset` only when os.sched_getaffinity is
        #        available (available in most of Linux system, but not OSX and Windows).
        #        When os.sched_getaffinity is not available, os.cpu_count() is called instead, but
        #        it doesn't repect cpuset.
        #        We don't take threading into account since each worker process is single threaded
        #        at this time.
        #
        #        We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
        #        other than `torch.set_num_threads` to 1 in the worker process, if the passing
        #        in functions use 3rd party modules that rely on those threading flags to determine
        #        how many thread to create (eg. numpy, etc), then it is caller's responsibility to
        #        set those flags correctly.
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):

            suggested_max_worker_msg = ((
                "Our suggested max number of worker in current system is {}{}, which is smaller "
                "than what this DataLoader is going to create.").format(
                    num_worker_suggest,
                    ("" if cpuset_checked else " (`cpuset` is not taken into account)"))
            ) if num_worker_suggest is not None else (
                "DataLoader is not able to compute a suggested max number of worker in current system.")

            warn_msg = (
                "This DataLoader will create {} worker processes in total. {} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary.").format(
                    num_worker_created,
                    suggested_max_worker_msg)
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satisfy mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))