#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`Normalizer`类，对数据进行正则化，支持`l1`,`l2`和`max`，默认为`l2`。
"""
import numpy as np
from sklearn import preprocessing


X = np.array([[1., -1.,  2.],
              [2.,  0.,  0.],
              [0.,  1., -1.]])

# 计算方法为：先计算每个样本的p范数，然后对该样本中的每个元素除以该范数
# 例如，对于样本[1., -1., 2.]，其l2范数为:
#   p = (|1.|^2 + |-1.|^2 + |2.|^2)^(1/2) = 6. ^ 0.5 = 2.449489...
# 则normalized之后为[1./p, -1./p, 2./p] = [0.40824829, -0.40824829, 0.81649658]
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)

normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
print(normalizer.transform(X))

# 转化新的数据
print(normalizer.transform([[-1., 1., 0.]]))
