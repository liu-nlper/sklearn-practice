## 预处理模块的使用

官方文档: [http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing "http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing")

### 1. 标准化

`std_mean_scaling.py`, Standardization, or mean removal and variance scaling.

#### 1.1 StandardScaler

将数据处理为均值为0，方差为1。

#### 1.2 MinMaxScaler

将特征压缩到一个范围内，默认范围为[0, 1]。

### 2. QuantileTransformer

`non_linear_transformation.py`，利用`QuantileTransformer`类，将数据转换到[0, 1]之间的均匀分布（也可以通过将`output_distribution`参数设置为`normal`，转换为正态分布）。

### 3. Normalization

利用`Normalizer`类，对数据进行正则化，支持`l1`,`l2`和`max`，默认为`l2`。

计算方法为：先计算每个样本的p范数，然后对该样本中的每个元素除以该范数。

例如，对于样本`[1., -1., 2.]`，其l2范数为:

    p = (|1.|^2 + |-1.|^2 + |2.|^2)^(1/2) = 6. ^ 0.5 = 2.449489...

则normalized之后为: \[1./p, -1./p, 2./p\] = \[0.40824829, -0.40824829, 0.81649658\]

### 4. Binarization
`binarization.py`，利用`Binarizer`类，将数据二值化。

通过`threshold`参数控制特征的取值，即小于等于`threshold`被转换为0，大于`threshold`的为1.

### 5. 处理分类特征
`categorical_features.py`，利用`OneHotEncoder`类处理分类特征 - `categorical_features.py`。

需要注意的是，若某个特征编号超出了训练数据的取值范围，则会出现以下错误:

    >>> enc.transform([[2, 1, 3]]).toarray()
    ...
    ValueError: unknown categorical feature present [2] during transform.

有两种处理方式，分别为

 (1) 将测试数据与训练数据合并之后，再`fit_transform`；

 (2) 将`OneHotEncoder`中的`handle_unknown`参数设置为`ignore`(默认为`error`)，则会忽略该特征，将对应位置置为0。

### 6. 处理缺失值

`missing_values.py`，利用`Imputer`类处理缺失值。


 用根据`strategy`缺失值，`strategy`共有3种取值，分别为:

    `mean`: 均值；
    `median`: 中位数；
    `most_frequent`: 众数。

### 7. 处理多项式特征

`polynomial_features.py`，`PolynomialFeatures`类中的`degree`参数控制多项式系数，若设置为2，则特征转换方式为:

    (x1, x2) -> (1, x1, x2, x1^2, x1*x2, x2^2)

例如：

    [1, 2] -> [1, 1, 2, 1, 2, 4]

### 8. 用户自定义处理方式

`function_transformer.py`，用户自定义预处理方式。

