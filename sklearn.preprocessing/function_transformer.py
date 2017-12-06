#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`FunctionTransformer`类，自定义处理方式。
"""
import numpy as np
from sklearn import preprocessing


transformer = preprocessing.FunctionTransformer(np.log1p)

X = np.array([[0, 1], [2, 3]])

print(transformer.transform(X))


# 自定义处理函数
def my_func(matrix):
    return matrix - np.max(matrix)

transformer = preprocessing.FunctionTransformer(my_func)

print(transformer.transform(X))
