r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer,CopulaGANSynthesizer,TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from collections import Counter
def databalancing(data=None,label_idx = -2,mode=None):
    if label_idx < 0:
        label_idx = data.shape[1]+label_idx     
    data.columns=data.columns.astype(str)
    if 'GAN' in mode:
        # 创建元数据对象并探测元数据信息
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        # 创建CTGANSynthesizer对象
        if mode=='CTGAN':
            model = CTGANSynthesizer(metadata, epochs=100, verbose=True)
        elif mode=='CopulaGAN':
            model = CopulaGANSynthesizer(metadata, epochs=100, verbose=True)

        # 训练模型
        model.fit(data)
        ser=data.iloc[:, label_idx].value_counts()
        num_rows=abs(ser[0]-ser[1])
        cnd=0
        if ser[0]>ser[1]:
            cnd=1
        condition = Condition({data.columns[label_idx]: cnd}, num_rows=num_rows)
        synthetic_train = model.sample_from_conditions(conditions=[condition])
        data=pd.concat([data,synthetic_train], axis=0)
        return data
    elif mode == 'TVAE':
        # 创建元数据对象并探测元数据信息
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        # 创建CTGANSynthesizer对象
        model = TVAESynthesizer(metadata, epochs=100)
        # 训练模型
        model.fit(data)
        ser=data.iloc[:, label_idx].value_counts()
        num_rows=abs(ser[0]-ser[1])
        cnd=0
        if ser[0]>ser[1]:
            cnd=1
        condition = Condition({data.columns[label_idx]: cnd}, num_rows=num_rows)
        synthetic_train = model.sample_from_conditions(conditions=[condition])
        data=pd.concat([data,synthetic_train], axis=0)
        return data
    elif mode == 'Undersampling':
        X = data.iloc[:,data.columns != data.columns[label_idx]]
        y = data.iloc[:,label_idx]
        undersampler = RandomUnderSampler()

        # 进行欠采样
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
    elif mode == 'Oversampling':
        X = data.iloc[:,data.columns != data.columns[label_idx]]
        y = data.iloc[:,label_idx]
        oversampler = RandomOverSampler()

        # 进行欠采样
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
    elif mode == 'SMOTE':
        X = data.iloc[:,data.columns != data.columns[label_idx]]
        y = data.iloc[:,label_idx]
        smote = SMOTE()

        # 进行合成抽样
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif mode == 'ADASYN': 
        X = data.iloc[:,data.columns != data.columns[label_idx]]
        y = data.iloc[:,label_idx]

        # 进行合成抽样
        X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    X_resampled_df = pd.DataFrame(X_resampled)
    y_resampled_df = pd.DataFrame(y_resampled)
    X_resampled_df.insert(loc=label_idx, column=y_resampled_df.columns[0], value=y_resampled_df)
    return X_resampled_df
def gaussian_noise(mean, std_dev, size):
    noise = np.random.normal(mean, std_dev, size)
    return noise
def uniform_noise(low, high, size):
    noise = np.random.uniform(low, high, size)
    return noise
def pulse_noise(feature_len,noise_len):
    noisy_indices = np.random.choice(feature_len, noise_len, replace=False)
    return noisy_indices
def get_statistics_dict(data):
    means = data.mean().values
    
    # 统计每列的标准差
    stds = data.std().values

    # 统计每列的最大值
    max_values = data.max().values

    # 统计每列的最小值
    min_values = data.min().values

    # 将结果放入字典
    statistics_dict = {
        "mean": means,
        "std": stds,
        "high": max_values,
        "low": min_values
    }
    return statistics_dict


