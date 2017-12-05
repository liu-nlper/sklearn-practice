#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    用bag of words表示文档，CountVectorizer类
"""
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stop_words = ['a', 'of', 'for']

# ngram_range控制n-gram的n，例如ngram_range=(1,2)即表示抽取uni-gram和bi-gram特征
# max_features控制词表的大小，若不为空，则按照词频进行排序后在作截取
count_vec = CountVectorizer(
    tokenizer=word_tokenize, stop_words=stop_words, max_features=100, ngram_range=(1, 2))
count_vec.fit(documents)

# count_vec对应的字典
count_vec.vocabulary_
print(len(count_vec.vocabulary_))

# 稀疏矩阵
sparse_matrix = count_vec.transform(documents)

# 转为稠密矩阵形式
matrix = sparse_matrix.toarray()
