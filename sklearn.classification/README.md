## 分类模型的使用

### 1. Logistic regression

[`logistic_regression.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/logistic_regression.py)，利用`logistic回归`处理多类别文本分类任务，参考文档：[sklearn文档](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py)

### 2. SVM
[`svm.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/svm.py)，利用`svm`处理多类别文本分类任务，可选的分类器包括`SVC`、`NuSVC`和`LinearSVC`，参考文档：[sklearn文档](http://scikit-learn.org/stable/modules/svm.html#multi-class-classification)

参数说明：

- C: 惩罚系数，即对误差的容忍度。C越大，说明越不能容忍错误，越容易过拟合；C越小，越容易欠拟合。
- gamma: gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。

### 3. KNN

[`knn.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/knn.py)，利用`knn`算法进行文本分类，参考文档：[`sklearn文档`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

可选用的算法包括：`ball_tree`、`kd_tree`和`brute`，该算法同样可用于相似`文档检索`任务。

### 4. Naive Bayes

[`naive_bayes.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/naive_bayes.py)，朴素贝叶斯算法在文本分类任务上的应用，参考文档：[`sklearn文档`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)

### 5. Ensemble methods

集成算法在文本分类上的应用。

#### 5.1 Random Forests

[`random_forests.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/random_forests.py)，参考文档：[`sklearn文档`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

**注**：值得注意的是，rf可以对特征的重要程度进行排序，可以用作特征选择。

#### 5.2 AdaBoost

[`ada_boost.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/ada_boost.py)，参考文档：[`sklearn文档`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)

### 6. Neural Network

#### 6.1 MLPClassifier

多层感知器在文本分类上的应用，参考文档：[`sklearn文档`](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)。

##### 6.1.1 [`mlp_classifier.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/mlp_classifier.py)

`mlp + tf-idf`，在20个类别18846篇文档上的`acc=0.807427`。

##### 6.1.2 [`mlp_classifier_lda.py`](https://github.com/liu-nlper/sklearn-practice/blob/master/sklearn.classification/mlp_classifier_lda.py)

`mlp + tf-idf + lda`，在20个类别18846篇文档上的`acc=0.807162`。
