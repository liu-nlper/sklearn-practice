#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`Imputer`类处理缺失值。
"""

import numpy as np
from sklearn import preprocessing

# 用根据`strategy`缺失值，`strategy`共有3种取值，分别为:
# `mean`: 均值；
# `median`: 中位数；
# `most_frequent`: 众数。
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

# 这里缺失值被处理为`(2 + 3 + 6) / 3 = 3.666...`
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

