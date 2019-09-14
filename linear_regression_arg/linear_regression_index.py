from linear_regression_arg.linear_regression import SimpleLinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

boston = datasets.load_boston()
x_train = boston.data[:,5] #rm 房屋数量
y_train = boston.target
x_train = x_train[y_train<50]
y_train = y_train[y_train<50]
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,random_state=666,test_size=0.2)
slr = SimpleLinearRegression()
slr.fit(x_train,y_train)
plt.scatter(x_train,y_train)
y_predict = slr.predict(x_test)
plt.plot(x_test,y_predict,color='r')
plt.title('y={k_}*x+{b_}'.format(k_=slr.k_,b_=slr.b_))
plt.show()
print(slr.r2_squared(x_test,y_test))

