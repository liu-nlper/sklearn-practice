#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`OneHotEncoder`类处理分类特征。
"""
from sklearn import preprocessing


enc = preprocessing.OneHotEncoder(handle_unknown='error')

# 注: `data`中每个数据共有三个特征，其中，第一个特征共有两种取值，取值为{0, 1}
# 第二个特征取值为{0, 1, 2, 3}(2并没有在特征中出现)，第三个特征取值为{0, 1, 2}
# 则特征的维度为`2+4+3=9`。
data = [[0, 0, 3], [1, 1, 0], [0, 3, 1], [1, 0, 2]]
enc.fit(data)

print(enc.transform([[2, 1, 3]]).toarray())

# 需要注意的是，若某个特征编号超出了训练数据的取值范围，则会出错，例如
# >>> enc.transform([[2, 1, 3]]).toarray()
# 会出现如下错误:
# ValueError: unknown categorical feature present [2] during transform.
# 有两种处理方式，分别为
# 1. 将测试数据与训练数据合并之后，再`fit_transform`
# 2. 将`handle_unknown`参数设置为`ignore`(默认为`error`)，则会忽略该特征，将对应位置置为0。
