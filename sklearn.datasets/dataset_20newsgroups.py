#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    加载20newsgroups数据
"""

from sklearn.datasets import fetch_20newsgroups


# 加载数据集
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# 3387 documents, 4 categories
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))

# list of str，每篇文章用一个字符串表示
documents = dataset.data

# list of str，每个label用一个字符串表示
labels = dataset.target
