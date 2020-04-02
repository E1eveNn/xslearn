# KNeighborsClassifier

```python
from sklearn.datasets import load_iris
# load dataset
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label'] = iris.target
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])

# display dataset
plt.scatter(df[:50]['sepal length (cm)'], df[:50]['sepal width (cm)'], label='-1')
plt.scatter(df[50:100]['sepal length (cm)'], df[50:100]['sepal width (cm)'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
```

![iris_-11](https://github.com/eLeVeNnN/xslearn/blob/master/examples/iris_-11.png)

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



KNeighborsClassifier(k=3, p=2)

- k:  nearest neighbors' numbers.
- p: use $L_p$ distance.

```python
from xslearn.models import KNeighborsClassifier

model = KNeighborsClassifier(k=3, p=2)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# print result in console
'''
fit complete -> time cost: 0.000000s
accuracy on test data is  1.0
'''
```

