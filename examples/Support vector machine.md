# Support vector machine

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, random_state=22)
y = [-1 if i == 0 else i for i in y]
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



SVC(kernel='linear', C=1, tol=1e-3,  degree=3, gamma='auto', beta=1, theta=-1, n_iter=10, verbose=2)

- kernel:  use kernel, you can pass 'linear'、'poly'、'rbf'、'laplace'、'sigmoid' and your own diy kernel.
- C: soft margin.
- tol: tolerance
- degree: poly kernel hyper-parameter.
- gamma: rbf and laplace kernel  hyper-parameter, 'auto' means gamma = 1 / length(dataset.features).
- beta: sigmoid kernel hyper-parameter.
- theta: sigmoid kernel hyper-parameter.
- n_iter: max training iterations.
- verbose: display training detail.

use `linear kernel`

```python
from xslearn.models import SVC
# use linear kernel for unlinear dataset.
model = SVC(kernel='linear', C=1, tol=1e-3, verbose=2, n_iter=100)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# print result in console
'''
iter 26 [=======>----------------------] -ETA 2s acc: 0.575, loss: 2.07736945670907
fit complete -> time cost: 0.493690s
accuracy on test data is  0.35
'''

```

use `sigmoid kernel`

```python
from xslearn.models import SVC
# use sigmoid kernel for unlinear dataset.
model = SVC(kernel='sigmoid', C=1, tol=1e-3, verbose=2, n_iter=100)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# print result in console
'''
iter 10 [===>--------------------------] -ETA 2s acc: 0.825, loss: 14.132350554806326
fit complete -> time cost: 0.421739s
accuracy on test data is  0.75
'''
```


