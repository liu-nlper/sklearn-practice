## 特征选择模块的使用

### 1. 词袋模型(CountVectorizer)
[`text2countvec.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.feature_extraction/text2countvec.py)，将文本表示成向量形式，其中`ngram_range`参数用于控制n-gram特征。

### 2. TFIDF(TfidfTransformer)
#### 2.1 [`text2tfidfvec.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.feature_extraction/text2tfidfvec.py)
`text2tfidfvec.py`，将文本映射到tf-idf向量空间。
#### 2.2 [`text2tfidfvec_pipeline.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.feature_extraction/text2tfidfvec_pipeline.py)
利用`Pipeline`类，将文本映射到tf-idf向量空间，整个过程是pipeline的，不需要一步步地转换。

### 3. LDA主题模型(LatentDirichletAllocation)
[`text2ldavec.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.feature_extraction/text2ldavec.py)，将文本映射到lda向量空间。