## 降维算法的使用
对预训练的词向量进行降维，`./data/wordvectors.txt`共包含词1w，词向量的维度为50。
以下实验均在`8核i7`处理器上进行。

官方文档: [http://scikit-learn.org/stable/modules/decomposition.html#decompositions](http://scikit-learn.org/stable/modules/decomposition.html#decompositions "http://scikit-learn.org/stable/modules/decomposition.html#decompositions")

### 1. [`pca_reduce_w2v_dim.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.decomposition/pca_reduce_w2v_dim.py) (pca降维算法)
以`苏州`为测试用词，分别输出降维前和降维之后与`苏州`语义最相近的10个词，可见降维效果比较好，降维之后找出的依然是与城市相关的词，降维耗时`0.4s`。
#### 降维前
    降维前，词向量维度为50，与`苏州`最相近的10个词:
   	0	东莞	0.800743
   	1	张家港	0.767392
   	2	营口	0.739155
   	3	盐城	0.737011
   	4	中山市	0.736084
   	5	外滩	0.733410
   	6	洛阳	0.733157
   	7	闵行	0.726662
   	8	石景山	0.704497
   	9	玉渊潭	0.700323
#### 降维后
    降维后，词向量维度为32，与`苏州`最相近的10个词:
   	0	东莞	0.704683
   	1	张家港	0.702397
   	2	营口	0.701115
   	3	洛阳	0.685188
   	4	德辉	0.677291
   	5	闵行	0.676041
   	6	外滩	0.672174
   	7	宋城	0.669389
   	8	中山市	0.667284
   	9	大观园	0.659511
### 2. [`fa_reduce_w2v_dim.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.decomposition/fa_reduce_w2v_dim.py) (Factor Analysis降维算法)
同样以`苏州`为测试用词，分别输出降维前和降维之后与`苏州`语义最相近的10个词，降维耗时`3.3s`。
#### 降维前
    降维前，词向量维度为50，与`苏州`最相近的10个词:
    0	东莞	0.800743
    1	张家港	0.767392
    2	营口	0.739155
    3	盐城	0.737011
    4	中山市	0.736084
    5	外滩	0.733410
    6	洛阳	0.733157
    7	闵行	0.726662
    8	石景山	0.704497
    9	玉渊潭	0.700323
#### 降维后
    降维后，词向量维度为32，与`苏州`最相近的10个词:
    0	东莞	0.739492
    1	石景山	0.684150
    2	外滩	0.678629
    3	营口	0.676477
    4	张家港	0.667767
    5	德辉	0.667334
    6	大观园	0.660639
    7	玉泉营	0.656201
    8	玉渊潭	0.656028
    9	盐城	0.649126
