#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`QuantileTransformer`类，将数据转换到[0, 1]之间的均匀分布。
"""
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)

# 转换训练数据
X_train_trans = quantile_transformer.fit_transform(X_train)

# array([ 4.3,  5.1,  5.8,  6.5,  7.9])
print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))

# array([ 0.00... ,  0.24...,  0.49...,  0.73...,  0.99... ])
print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))

# 转换新的数据
X_test_trans = quantile_transformer.transform(X_test)
# array([ 4.4  ,  5.125,  5.75 ,  6.175,  7.3  ])
print(np.percentile(X_test[:, 0], [0, 25, 50, 75, 100]))
# array([ 0.01...,  0.25...,  0.46...,  0.60... ,  0.94...])
print(np.percentile(X_test_trans[:, 0], [0, 25, 50, 75, 100]))
