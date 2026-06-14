r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""
import numpy as np
import pandas as pd
from .crop import crop_trace,crop_adj
import os
class _TraceDatasetReader:
    def __init__(self, filenames):
        self._filenames = filenames
    def fetch(self, possibly_filename_index,max_len,drop_rate,scaler):
        gnssfilename=self._filenames['gnss'][possibly_filename_index] if self._filenames['gnss'] is not None else None
        adjfilename=self._filenames['adj'][possibly_filename_index] if self._filenames['adj'] is not None else None
        index = self._filenames['index'][possibly_filename_index]
        data = pd.read_excel(gnssfilename, header=0)
        n=len(data)
        if index is not None:
            data = data.iloc[index]
        # 将数据转换为 float32 类型的 numpy 数组
        points = np.array(data).astype('float32')
        coordinates = points[:,-3:-1]
        if scaler is not None:
            # 提取除了最后一列之外的所有列
            data_to_normalize = points[:, :-1]

            # 对数据进行归一化
            normalized_data = scaler.fit_transform(data_to_normalize)

            # 将归一化的数据和最后一列合并
            points = np.hstack((normalized_data, points[:, -1].reshape(-1, 1)))
        cropped_points = crop_trace(points,max_len,drop_rate)
        cropped_coordinates = crop_trace(coordinates,max_len,drop_rate)
        trace_id = gnssfilename
        if adjfilename is not None:
            indices = np.load(adjfilename)
            # 使用布尔索引恢复方阵
            adj = np.zeros((n, n), dtype=int).astype('float32')
            adj[indices[:,0], indices[:, 1]] = 1.0
            if index is not None:
                adj = adj[index][:,index]
            cropped_adjs=crop_adj(adj,max_len,drop_rate)
        try:
                return (cropped_points ,trace_id ,cropped_adjs,cropped_coordinates)
        except:
                return (cropped_points ,trace_id,cropped_coordinates)
class _PointDatasetReader:
    def __init__(self, filenames):
        self._filenames = filenames
    def fetch(self, possibly_filename_index,scaler):
        gnssfilename=self._filenames['gnss'][possibly_filename_index] if self._filenames['gnss'] is not None else None
        index = self._filenames['index'][possibly_filename_index]
        gnss = pd.read_excel(gnssfilename, header=0)
        if scaler is not None:
            # 提取除最后一列之外的所有列
            columns_to_normalize = gnss.columns[:-1]

            # 对数据进行归一化
            gnss[columns_to_normalize] = scaler.fit_transform(gnss[columns_to_normalize])
        n=gnss.shape[0]
        if index is not None:
            gnss = gnss.iloc[index]
        data = dict(
            gnss = gnss,
            trace_id = os.path.basename(gnssfilename),
            adj = None,
            adj_id = None
        )
        if self._filenames['adj'] is not None:
            adjfilename=self._filenames['adj'][possibly_filename_index]
            indices = np.load(adjfilename)
            
            # 使用布尔索引恢复方阵
            adj = np.zeros((n, n), dtype=int).astype('float32')
            adj[indices[:, 0], indices[:, 1]] = 1.0
            if index is not None:
                adj = adj[index][:,index]
            data = dict(
                gnss = gnss,
                trace_id = os.path.basename(gnssfilename),
                adj = adj,
                adj_id = os.path.basename(adjfilename),
            )
        return data
