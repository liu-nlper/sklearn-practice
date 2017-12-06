#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`PolynomialFeatures`类，处理多项式特征。
"""
import numpy as np
from sklearn import preprocessing

X = np.arange(6).reshape(3, 2)
# array([[0, 1],
#       [2, 3],
#       [4, 5]])

# `degree`参数控制多项式系数，若设置为2，则特征转换方式为:
# (x1, x2) -> (1, x1, x2, x1^2, x1*x2, x2^2)
poly = preprocessing.PolynomialFeatures(degree=2)

print(poly.fit_transform(X))
