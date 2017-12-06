#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    MLPClassifier算法在文本分类任务上的应用
"""

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# 加载数据集。20个类别共18846个实例
print('加载数据集...')
dataset = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)
X_ = dataset.data
y = dataset.target

# 类别数
nb_classes = np.unique(y).shape[0]

# 将训练数据映射到tf-idf空间
print('将数据映射到tf-idf向量空间...')
pipeline = Pipeline(steps=[
    ('vect', CountVectorizer(stop_words='english', tokenizer=word_tokenize, max_df=0.6, min_df=0.01)),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(X_).toarray()

# 划分训练集、测试集
print('划分训练集和测试集...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.2)

# 建立模型，其中`validation_fraction`为开发集的比例
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256,), activation='relu', solver='adam',
    batch_size=256, early_stopping=True, validation_fraction=0.2,
    alpha=0.0001, random_state=123)

# 训练
print('训练模型...')
mlp_model.fit(X_train, y_train)

# 预测
print('预测...')
y_pred_train = mlp_model.predict(X_train)
y_pred_test = mlp_model.predict(X_test)

# 计算acc
acc_train = np.sum(y_train == y_pred_train) / y_train.shape[0]
acc_test = np.sum(y_test == y_pred_test) / y_test.shape[0]

print('acc of train: %f' % acc_train)  # 0.882728
print('acc of test: %f' % acc_test)  # 0.795756
