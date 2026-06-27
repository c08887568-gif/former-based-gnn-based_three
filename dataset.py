from  fieldroaddatapipeline.dataset import Dataset
from  fieldroaddatapipeline.datareader import FieldRoadDataReader
from  fieldroaddatapipeline.dataaugmenter import Compose,DataBalancer,Gaussian_Noise,Uniform_Noise,Pulse_Noise,get_statistics_dict
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
__all__ = [
    "TraceDataset",
    "PointDataset",
]
class GraphDataset(Dataset):
    def __init__(self, path, mode='train', num_workers = 0,max_len = 5000, drop_rate = 0.01):
        assert mode in ['train', 'valid', 'test','predict'], 'mode is one of train, valid ,test.'
        self.path = path
        self.mode = mode
        self.num_workers = num_workers
        self.max_len = max_len
        self.drop_rate = drop_rate
        self.data = []
        self.adjs = {}

        if mode in ['train','valid','test','predict'] :
            fileiter=FieldRoadDataReader(path,dataset_format='json',mode = 'Trace',num_workers = num_workers,max_len = max_len, drop_rate = drop_rate,scaler = MinMaxScaler())
            for cropped_points,trace_id,cropped_adjs,cropped_coordinates in fileiter:
                self.adjs[str(trace_id)] = cropped_adjs
                for points,adj,coordinates in zip(cropped_points,cropped_adjs,cropped_coordinates):
                    nodes = points[:,:-1] 
                    labels = points[:,-1].reshape(-1,1)
                    self.data.append((nodes, labels, trace_id, adj,coordinates))
        else:
            fileiter=FieldRoadDataReader(path,dataset_format='json',mode = 'Trace',num_workers = num_workers,max_len = max_len, drop_rate = drop_rate,scaler = MinMaxScaler())
            for cropped_points,trace_id,cropped_adjs,cropped_coordinates in fileiter:
                self.adjs[str(trace_id)] = cropped_adjs
                for points,adj,coordinates in zip(cropped_points,cropped_adjs,cropped_coordinates): 
                    self.data.append((points,trace_id, adj,coordinates))

    def __getitem__(self, index):
        if self.mode in ['train','valid','test']:
            points, labels, trace_id , adj,coordinates = self.data[index]
            return points, labels, adj, trace_id
        elif self.mode == 'predict':
            points, labels, trace_id , adj,coordinates = self.data[index]
            return points, labels, adj, trace_id,coordinates
        else:
            points, trace_id ,adj ,coordinates = self.data[index]
            return points, adj, coordinates
    def __getadj__(self, traid):
        return self.adjs[str(traid)]

    def __len__(self):
        return len(self.data)
    def __str__(self):
        return (
            f"GraphDataset(\n"
            f"  Path: {self.path}\n"
            f"  Mode: {self.mode}\n"
            f"  Num_workers: {self.num_workers}\n"
            f"  Max_len: {self.max_len}\n"
            f"  Drop_rate: {self.drop_rate}\n"
            f")"
        )

class CachedGraphDataset(Dataset):
    def __init__(self, cache_path, mode='train', return_coordinates=False):
        assert mode in ['train', 'valid', 'test', 'predict'], 'mode is one of train, valid ,test, predict.'
        self.cache_path = cache_path
        self.mode = mode
        self.return_coordinates = return_coordinates
        self.data = torch.load(cache_path, map_location='cpu')

    def __getitem__(self, index):
        item = self.data[index]
        if self.mode == 'predict' or self.return_coordinates:
            if 'coordinates' not in item:
                raise ValueError("AUX_COORDINATES_NOT_FOUND")
            return item['points'], item['labels'], item['edge_index'], item['trace_id'], item['coordinates']
        return item['points'], item['labels'], item['edge_index'], item['trace_id']

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return (
            f"CachedGraphDataset(\n"
            f"  Cache: {self.cache_path}\n"
            f"  Mode: {self.mode}\n"
            f"  Return coordinates: {self.return_coordinates}\n"
            f"  Samples: {len(self.data)}\n"
            f")"
        )
    
