# RandomForestClassifier

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, random_state=22)
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
# (80, 20)
# (20, 20)
# (80,)
# (20,)

```



RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, max_samples=None, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, verbose=0, random_state=None)

- n_estimators:  base estimator's numbers.
- criterion: cart tree's split method.
- max_depth: cart tree's max depth.
- max_samples: sampling ratio, None denotes sampling all the dataset; float number denotes sampling ratio of all dataset, which should be in [0., 1.]; int number denotes sampling number.
- max_features: samling feature ratio, 'auto' denotes sampling feature numbers k = $log_2d$, where d denotes all feature's numbers; you can also specify this parameter as float or int.
- max_leaf_nodes: cart tree's max leaf node numbers.
- bootstrap: whether use bootstrap sampling.
- oob_score: use out-of-bag datas.
- n_jobs: use multi-thread; -1 equals to 1.
- verbose: display training detail.
- random_state: random seed.



compare `RandomForestClassifier`  with `DecisionTreeClassfier`



```python
from xslearn.models import DecisionTreeClassifier

model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
model1.score(X_test, y_test)
# print result in console
'''
fit complete -> time cost: 0.629613s
accuracy on test data is  0.85
'''

```



```python
from xslearn.models import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators=5)
model2.fit(X_train, y_train)
model2.score(X_test, y_test)
# print result in console
'''
fit complete -> time cost: 0.208873s
fit complete -> time cost: 0.114928s
fit complete -> time cost: 0.223865s
fit complete -> time cost: 0.223862s
fit complete -> time cost: 0.127921s
accuracy on test data is  0.9
'''
```



**accuracy on test data raised from 0.85 to 0.9**

We can use **multi-thread** to speed up training:

```python
from xslearn.models import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators=5, n_jobs=8)
model2.fit(X_train, y_train)
model2.score(X_test, y_test)
```





