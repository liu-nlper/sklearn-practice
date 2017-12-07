#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    计算非负特征与类别之间的卡方统计量，并在文本分类任务上进行检验。
"""

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression


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
    ('vect', CountVectorizer(stop_words='english', tokenizer=word_tokenize, max_df=0.5, min_df=0.01)),
])
X = pipeline.fit_transform(X_)


# 降维前...
print('特性选择前...')

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

print('acc: %f\n\n' % acc)  # 0.817109


# 特性选择后...
print('特性选择后...')

# 特征选择
chi2 = SelectKBest(chi2, k=512)
X_train = chi2.fit_transform(X_train, y_train)
X_test = chi2.transform(X_test)

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

print('acc: %f\n\n' % acc)  # 0.820059