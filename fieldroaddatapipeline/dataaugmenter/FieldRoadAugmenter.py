r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""
from .functional import (databalancing,
                        gaussian_noise,
                        uniform_noise,
                        pulse_noise,)
import random
import numpy as np
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
class DataBalancer:
    def __init__(self, mode):
        assert mode in ['CTGAN','CopulaGAN','TVAE','Undersampling','Oversampling','SMOTE','ADASYN']
        self.mode = mode

    def __call__(self, data,label_idx = -2):
        resampled_data=databalancing(data,label_idx,self.mode)
        return resampled_data        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Gaussian_Noise:
    def __init__(self,p,mean,std):
        self.p = p
        self.mean = mean
        self.std = std
    def __call__(self, data):
        if data.shape[0]==1 or data.ndim == 1:
            if random.random() < self.p:
                noise_data = data + gaussian_noise(self.mean,self.std,data.shape)
            else:
                noise_data = data
        else:
            noise_len = int(data.shape[0] * self.p)
            noisy_indices = np.random.choice(data.shape[0], noise_len, replace=False)
            noise_data = data.copy()
            noise_data[noisy_indices,:] += gaussian_noise(self.mean,self.std,(noisy_indices.shape[0],data.shape[1]))
            print(noisy_indices.shape[0])
            
        return noise_data
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Uniform_Noise:
    def __init__(self,p,low,high):
        self.p = p
        self.low = low
        self.high = high
    def __call__(self, data):
        if data.shape[0]==1 or data.ndim == 1:
            if random.random() < self.p:
                noise_data = data + uniform_noise(self.low,self.high,data.shape)
            else:
                noise_data = data
        else:
            noise_len = int(data.shape[0] * self.p)
            noisy_indices = np.random.choice(data.shape[0], noise_len, replace=False)
            noise_data = data.copy()
            noise_data[noisy_indices,:] += uniform_noise(self.low,self.high,(noisy_indices.shape[0],data.shape[1]))
        return noise_data
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Pulse_Noise:
    def __init__(self,p,pulse_ratio,pulse_noise_value):
        self.p = p
        self.pulse_ratio=pulse_ratio
        self.pulse_noise_value = pulse_noise_value
    def __call__(self, data):
        if data.shape[0]==1 or data.ndim == 1:
            if random.random() < self.p:
                feature_len = data.shape[0]
                noise_len = int(data.shape[0] * self.pulse_ratio)
                noisy_indices = pulse_noise(feature_len,noise_len)
                noise_data = data.copy()
                noise_data[noisy_indices] = self.pulse_noise_value
            else:
                noise_data = data
        else:
            noise_len = int(data.shape[0] * self.p)
            noisy_indices = np.random.choice(data.shape[0], noise_len, replace=False)
            noise_data = data.copy()
            noise_data[noisy_indices,:] = self.pulse_noise_value
        return noise_data
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"