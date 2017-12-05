#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    用bag of words表示文档，CountVectorizer类
"""
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

count_vec = CountVectorizer(tokenizer=word_tokenize, stop_words=stop_words)
sparse_matrix_count = count_vec.fit_transform(documents)

tfidf_vec = TfidfTransformer()
tfidf_vec.fit(sparse_matrix_count)
tfidf_vec.transform(sparse_matrix_count)

sparse_matrix_tfidf = tfidf_vec.transform(sparse_matrix_count)
print(sparse_matrix_tfidf)

# 将一个新的文档映射到tfidf空间
new_doc = "The generation of random xxx"
new_doc_count = count_vec.transform([new_doc])
new_doc_tfidf = tfidf_vec.transform(new_doc_count)
print(new_doc_tfidf)

# 转为稠密矩阵
new_doc_tfidf.toarray()
