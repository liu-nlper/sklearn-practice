#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    朴素贝叶斯算法在文本分类任务上的应用
"""

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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

# 划分训练集、测试集
print('划分训练集和测试集...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.2)

# 建立模型
gnb_model = GaussianNB()

# 训练
print('训练模型...')
gnb_model.fit(X_train, y_train)

# 预测
print('预测...')
y_pred_train = gnb_model.predict(X_train)
y_pred_test = gnb_model.predict(X_test)

# 计算acc
acc_train = np.sum(y_train == y_pred_train) / y_train.shape[0]
acc_test = np.sum(y_test == y_pred_test) / y_test.shape[0]

print('acc of train: %f' % acc_train)  # 0.974529
print('acc of test: %f' % acc_test)  # 0.871681
