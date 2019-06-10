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
print(X.shape)
#lets plot the linear curve using linear regression model
from  sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

#lets test for a polynomial model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = Pipeline((
	("poly_features",PolynomialFeatures(degree=2, include_bias=False)),
	("lin_reg", LinearRegression()),
))
plot_learning_curves(polynomial_regression, X, y)
