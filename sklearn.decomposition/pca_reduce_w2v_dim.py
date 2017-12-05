#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    利用PCA算法降低词向量的维度
"""

import numpy as np
from sklearn.decomposition import PCA
from gensim.models.keyedvectors import KeyedVectors


# load word vectors
print('加载词向量...')
path_vec = './data/wordvectors.txt'
word_vectors = KeyedVectors.load_word2vec_format(path_vec, binary=False)

words = word_vectors.vocab
word_dim = 50
word_matrix = np.zeros((len(words), word_dim), dtype='float32')
for i, word in enumerate(words):
    word_matrix[i, :] = word_vectors[word]

# 利用pca降维词向量
print('利用pca算法降维...')
new_word_dim = 32
word_matrix_post = PCA(new_word_dim).fit_transform(word_matrix)

# 降维后的词向量写入文件
path_vec_new = './data/wordvectors_new.txt'
file_w = open(path_vec_new, 'w', encoding='utf-8')
file_w.write('%s %s\n' % (len(words), new_word_dim))
for i, word in enumerate(words):
    file_w.write('%s %s\n' % (word, ' '.join(list(map(lambda d: '%.8f' % d, word_matrix_post[i])))))
file_w.close()

# 加载降维后的词向量
print('加载降维之后的词向量...')
word_vectors_new = KeyedVectors.load_word2vec_format(path_vec_new, binary=False)

# 找最相近的词
word_test = '苏州'
topn = 10
print('\n降维前，词向量维度为%d，与`%s`最相近的%d个词:' % (word_vectors.vector_size, word_test, topn))
for i, word in enumerate(word_vectors.most_similar(word_test, topn=topn)):
    print('\t%d\t%s\t%f' % (i, word[0], word[1]))

print('\n降维后，词向量维度为%d，与`%s`最相近的%d个词:' % (word_vectors_new.vector_size, word_test, topn))
for i, word in enumerate(word_vectors_new.most_similar(word_test, topn=topn)):
    print('\t%d\t%s\t%f' % (i, word[0], word[1]))
