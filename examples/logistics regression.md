# Logistic Regression

```python
from sklearn.datasets import load_iris
# load dataset
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label'] = iris.target
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]

# display dataset
plt.scatter(X[:50, 0],X[:50, 1], label='0')
plt.scatter(X[50:, 0],X[50:, 1], label='1')
plt.legend()
plt.show()
```

![iris_01](D:%5CPythonProjects%5Cxslearn_package%5Cxslearn%5Cexamples%5Ciris_01.png)

```python
from xslearn.utils import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# output:
# (80, 2)
# (20, 2)
# (80,)
# (20,)

```



LogisticRegression(n_features, lr=0.1, n_iter=1000, shuffle=False, verbose=1)

- n_features: training datasets' feature numbers, for example, training dataset shape is (20, 3), in this condition,  n_features=3.
- lr: learning rate.
- n_iter: max iteration numbers.
- shuffle: whether shuffle training data.
- verbose: print/ no print training details.

```python
from xslearn.models import LogisticRegression

model = LogisticRegression(n_features=2, n_iter=1000, verbose=2)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# print result in console
'''
iter 1000 [==============================>] -ETA 0us acc: 0.9875, loss: 0.09939489423951078
fit complete -> time cost: 0.441735s
accuracy on test data is  1.0
'''
```

use $model.summary()$ to see training details.



<img src="D:%5CPythonProjects%5Cxslearn_package%5Cxslearn%5Cexamples%5Clr_train.png" alt="lr_train" style="zoom:200%;" />