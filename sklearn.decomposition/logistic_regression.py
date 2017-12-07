#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    用降维算法对特征进行降维，并利用logistic回归验证降维效果。
"""

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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
X = pipeline.fit_transform(X_).toarray()
print(X.shape)

# 设置降维后的特征维度
print('降维...')
feature_dim = 512
X = PCA(feature_dim).fit_transform(X)  # shape=(3387, 512)

# 划分训练集、测试集
print('划分训练集和测试集...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.2)

# 建立模型
lr_model = LogisticRegression(
    solver='saga', multi_class='multinomial', penalty='l2',
    fit_intercept=True, max_iter=5, random_state=42)

# 训练
print('训练模型...')
lr_model.fit(X_train, y_train)

# 预测
print('预测...')
y_pred = lr_model.predict(X_test)

# 计算acc
acc = np.sum(y_test == y_pred) / y_test.shape[0]

print('acc: %f' % acc)  # 0.926254
