r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""
import numpy as np
import pandas as pd
import os
import shutil
import json
class _TraceDatasetWriter:
    def __init__(self, data ,path,mode,data_type,filenames):
        self._data = data
        self._mode = mode
        self._data_type = data_type
        self._filenames = filenames
        self._path_gnss = os.path.join(os.path.dirname(path['gnss']), path['gnss'].split('/')[-1]+'_'+ mode +'_'+ data_type)
        self._path_adj = os.path.join(os.path.dirname(path['adj']), path['adj'].split('/')[-1]+'_'+ mode +'_'+ data_type)
        os.makedirs(self._path_gnss, exist_ok=True)
        os.makedirs(self._path_adj, exist_ok=True)
    def write(self, index):
        filename = self._filenames['gnss'][index].split('/')[-1]
        filename_prefix = filename[:-5]
        gnss = pd.DataFrame(self._data[index]['gnss'])
        try:
            gnss['time'] = pd.to_datetime(gnss['time'],format='mixed')
            gnss['time'] = pd.to_datetime(gnss['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            try:
                gnss['时间'] = pd.to_datetime(gnss['时间'],format='mixed')
                gnss['时间'] = pd.to_datetime(gnss['时间']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        gnss_outputname = self._path_gnss + '/' + filename_prefix +'.xlsx'
        gnss.to_excel(gnss_outputname, index=False)
        
        if self._data[index]['adj'] is not None:
            adj = self._data[index]['adj']
            adj_outputname = self._path_adj + '/' + filename_prefix +'.npy'
            np.save(adj_outputname,adj)
            return gnss_outputname + '保存成功'+'\n' + adj_outputname + '保存成功'
        
        return gnss_outputname + '保存成功'
