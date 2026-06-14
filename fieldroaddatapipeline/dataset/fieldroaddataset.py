from  . import Dataset
from  ..datareader import FieldRoadDataReader
from ..dataaugmenter import Compose,DataBalancer,Gaussian_Noise,Uniform_Noise,Pulse_Noise,get_statistics_dict
import numpy as np
import pandas as pd
__all__ = [
    "TraceDataset",
    "PointDataset",
]
class TraceDataset(Dataset):
    def __init__(self, path, mode='train', num_workers = 0,max_len = 5000, drop_rate = 0.01):
        assert mode in ['train', 'valid', 'test'], 'mode is one of train, eval ,test.'
        self.mode = mode
        self.data = []
        self.adjs = {}

        if mode == 'train' or mode == 'valid':
            fileiter=FieldRoadDataReader(path,dataset_format = 'Trace',num_workers = num_workers,max_len = max_len, drop_rate = drop_rate)
            for cropped_points,cropped_labels,trace_id,cropped_adjs in fileiter:
                self.adjs[str(trace_id)] = cropped_adjs
                for points,labels,adj in zip(cropped_points,cropped_labels,cropped_adjs): 
                    self.data.append((points, labels, trace_id, adj))
                
        else:
            fileiter=FieldRoadDataReader(path,dataset_format = 'Trace',num_workers = num_workers,max_len = max_len, drop_rate = drop_rate)
            for cropped_points,trace_id,cropped_adjs in fileiter:
                self.adjs[str(trace_id)] = cropped_adjs
                for points,adj in zip(cropped_points,cropped_adjs): 
                    self.data.append((points,trace_id, adj))

    def __getitem__(self, index):
        if self.mode in ['train', 'valid']:
            points, labels, trace_id , adj= self.data[index]
            return points, labels, adj
        else:
            points, trace_id ,adj= self.data[index]
            return points, adj ,trace_id

    def __getadj__(self, traid):
        return self.adjs[str(traid)]

    def __len__(self):
        return len(self.data)
    
class PointDataset(Dataset):
    def __init__(self,path, mode='train', num_workers = 0):
        assert mode in ['train', 'valid', 'test'], 'mode is one of train, eval ,test.'
        self.mode = mode
        self.data = []
        self.statistics_dict = {}
        self.transform = None
        if mode == 'train'or mode == 'valid':
            train_data = pd.DataFrame()
            fileiter = FieldRoadDataReader(path,dataset_format = 'Point',num_workers = num_workers)
            for subdata in fileiter:
                train_data = pd.concat([train_data, subdata], ignore_index=True)
            balancer = DataBalancer(mode='CopulaGAN')
            train_data = balancer(train_data,-2)
            self.statistics_dict = get_statistics_dict(train_data.iloc[:,:-2])
            training_dataSet = list(np.array(train_data)[:, :43])
            label = train_data.iloc[:, 43]
            trace_id = list(train_data.iloc[:, 44])
            self.data = [(point.astype('float32'), np.array([label]).astype('int64'), trace_id) for point, label, trace_id in
                         zip(training_dataSet, label, trace_id)]
            self.transform = Compose([Pulse_Noise(0.5,0.5,100),
                                      Gaussian_Noise(0.2,self.statistics_dict['mean'],self.statistics_dict['std']),
                                      Uniform_Noise(0.2,self.statistics_dict['low'],self.statistics_dict['high'])])
        else:
            train_data = pd.DataFrame()
            fileiter = FieldRoadDataReader(path,dataset_format = 'Point',num_workers = num_workers)
            for subdata in fileiter:
                eval_data = pd.concat([eval_data, subdata], ignore_index=True)
            evaling_dataSet = list(np.array(eval_data)[:, :43])
            trace_id = list(eval_data.iloc[:, 43])
            self.data = [(point.astype('float32'), trace_id) for point, label, trace_id in
                         zip(evaling_dataSet, trace_id)]
    def __getitem__(self, index):
        if self.mode in ['train', 'valid']:
            point, label, trace_id = self.data[index]
            point = self.transform(point)
            return point,label
        else:
            point, trace_id = self.data[index]
            point = self.transform(point)
            return point,trace_id
    def __len__(self):
        return len(self.data)