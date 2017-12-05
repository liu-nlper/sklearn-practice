#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用`Pipeline`类，将文本映射到tf-idf向量空间，整个过程是pipeline的，不需要一步步地转换。
"""
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

documents = ["Human machine interface for lab abc computer applications.",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stop_words = ['a', 'of', 'for']

pipeline = Pipeline(steps=[
    ('vect', CountVectorizer(stop_words=stop_words, tokenizer=word_tokenize)),
    ('tfidf', TfidfTransformer()),
])

# 直接将documents映射到tfidf空间，不需要中间过程
sparse_matrix_tfidf = pipeline.fit_transform(documents)

# 将一个新的文档映射到tfidf空间
new_doc = "The generation of random xxx"
new_doc_tfidf = pipeline.transform([new_doc])
print(new_doc_tfidf)

# 转为稠密矩阵
print(new_doc_tfidf.toarray())
