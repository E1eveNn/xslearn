# DecisionTreeClassifier

```python
def create_data():
    X = [['青年', '否', '否', '一般'],
               ['青年', '否', '否', '好'],
               ['青年', '是', '否', '好'],
               ['青年', '是', '是', '一般'],
               ['青年', '否', '否', '一般'],
               ['中年', '否', '否', '一般'],
               ['中年', '否', '否', '好'],
               ['中年', '是', '是', '好'],
               ['中年', '否', '是', '非常好'],
               ['中年', '否', '是', '非常好'],
               ['老年', '否', '是', '非常好'],
               ['老年', '否', '是', '好'],
               ['老年', '是', '否', '好'],
               ['老年', '是', '否', '非常好'],
               ['老年', '否', '否', '一般'],
               ]
   	y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况']
    return X, y, labels

X, y, attr_sets = create_data()
```

use $xslearn.utils.train\_test\_split()$ to split data

```python
from xslearn.utils import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# output:
# (12, 4)
# (3, 4)
# (12,)
# (3,)

```



DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, attr_sets=None)

- criterion:  split method,  we support 'entropy'、'entropy_ratio'、'gini'.
- max_depth: generate tree's max depth.
- max_features: generate tree's max split feature numbers.
- max_leaf_nodes: generate tree's max leaf node numbers.
- attr_sets: dataset's attribute name.

```python
from xslearn.models import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini', attr_sets=attr_sets)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# print result in console
'''
fit complete -> time cost: 0.002997s
accuracy on test data is  1.0
'''

```

use $model.visualize()$ to see the constructed tree.

```python
model.visualize()
```

<img src="D:%5CPythonProjects%5Cxslearn_package%5Cxslearn%5Cexamples%5Cdt.png" alt="dt" style="zoom:200%;" />

