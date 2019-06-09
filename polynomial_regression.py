import numpy as np
import matplotlib.pyplot as plt
#sample dataset
m = 100
X = 6  * np.random.rand(m,1) -3
y = 0.5 * X ** 2 + X + np.random.randn(m,1)

'''
plt.scatter(X,y,c="blue")
plt.show()'''

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])

#now we can imploy the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
print(lin_reg.intercept_,lin_reg.coef_)

#now lets see how the model fits into the dataset
plt.plot(X, lin_reg.predict(X_poly),color='g')
plt.scatter(X,y,c="blue")
plt.show()




