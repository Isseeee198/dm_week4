#import datasets
#X, Y = datasets.load_linear_example1()

import datasets2
X, Y = datasets2.load_nonlinear_example1()
#ex_X = datasets2.polynomial2_features(X)
ex_X = datasets2.polynomial3_features(X)
#print(ex_X)
#print(Y)


import regression
model = regression.LinearRegression()

import importlib
importlib.reload(regression)
#model = regression.LinearRegression()
model = regression.RidgeRegression(alpha=0.1)
#model.alpha
model.fit(ex_X,Y)
print(model.theta)

#model.fit(X,Y)
#model.theta
#model.predict(X)
#model.score(X,Y)
#print(model.score(X,Y))