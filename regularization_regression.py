from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(model, X,y):
	X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)
	train_errors, val_errors =[], []
	for m  in range(1, len(X_train)):
		model.fit(X_train[:m], y_train[:m])
		y_train_predict = model.predict(X_train[:m])
		y_val_predict = model.predict(X_val)
		train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
		val_errors.append(mean_squared_error(y_val_predict, y_val))
	plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
	plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
	plt.show()

#sample dataset
m = 100
X =  2  * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)
#print(X.shape)
#lets perform Ridge regressio with sklearn using a matrix factorization technique by cholesky
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X,y)
print(ridge_reg.predict([[1.5]]))

#Ridge regression using stochastic gradient descent
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty = "l2")
sgd_reg.fit(X,y.ravel())
print(sgd_reg.predict([[1.5]]))

#lets perform Lasso regression with sklearn 
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
print(lasso_reg.predict([[1.5]]))

#lets perform ElasticNet with sklearn its the mix of the two Lasso ad Ridge
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X,y)
print(elastic_net.predict([[1.5]]))

