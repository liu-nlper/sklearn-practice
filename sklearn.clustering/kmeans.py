#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用logistic回归处理多类别文本分类任务。
"""

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn import metrics


# 加载数据集。四个类别共3387个实例
print('加载数据集...')
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
X_ = dataset.data
y = dataset.target
# 类别数
nb_classes = np.unique(y).shape[0]

# 将训练数据映射到tf-idf空间
print('将数据映射到tf-idf向量空间...')
pipeline = Pipeline(steps=[
    ('vect', CountVectorizer(stop_words='english', tokenizer=word_tokenize, max_df=0.4, min_df=0.01)),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(X_)

# 建立模型
km_model = KMeans(n_clusters=nb_classes, init='k-means++', max_iter=100, n_init=1)

km_model.fit(X)

print(y[:10])
print(km_model.labels_[:10])

# 聚类聚类算法的几个评价指标
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, km_model.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(y, km_model.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(y, km_model.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(y, km_model.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km_model.labels_, sample_size=1000))
