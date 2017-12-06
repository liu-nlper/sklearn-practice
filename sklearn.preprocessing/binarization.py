#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`Binarizer`类，将数据二值化。
"""
import numpy as np
from sklearn import preprocessing


X = np.array([[1., -1.,  2.],
              [2.,  0.,  0.],
              [0.,  1., -1.]])

binarizer = preprocessing.Binarizer(threshold=0.).fit(X)  # fit does nothing

print(binarizer.transform(X))
