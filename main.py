import datasets
X, Y = datasets.load_linear_example1()


import regression
model = regression.LinearRegression()

import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
#model.theta
model.predict(X)
print(model.predict(X))