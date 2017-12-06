#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Standardization, or mean removal and variance scaling.
"""
import numpy as np
from sklearn import preprocessing

X_train = np.array([[1., -1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., -1.]])


# 1. scale

# scale，将数据处理为均值为0，方差为1
X_scaled = preprocessing.scale(X_train)
print(np.std(X_train, axis=0))
print(X_scaled)
print(X_scaled.mean(axis=0))  # [0., 0., 0.]
print(X_scaled.std(axis=0))   # [1., 1., 1.]

# StandardScaler
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler.mean_)   # 均值，[ 1., 0., 0.33333333]
print(scaler.scale_)  # 标准差，[0.81649658, 0.81649658, 1.24721913]

# 计算公式：(X - mean) / std
X_scaled = scaler.transform(X_train)
print(X_scaled)

# 转换新的数据，使用在训练数据上计算得出的均值和方差
# mean = [ 1., 0., 0.33333333]
# std = [0.81649658, 0.81649658, 1.24721913]
X_test = [[-1., 1., 0.]]
# (X_test - mean) / std
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)

# 将特征压缩到一个范围内，默认区间为[0, 1]
# The motivation to use this scaling include robustness to very small standard
# deviations of features and preserving zero entries in sparse data.
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)
# 新的数据
X_test = np.array([[-3., -1., 4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print(X_test_minmax)
