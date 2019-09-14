__author__ = 'Guan'
import  numpy as np
from math import sqrt
#回归问题
#最小二乘法

class SimpleLinearRegression():
    '''
    SimpleLinearRegression only can slove one dimension
    '''
    def __init__(self):
        self.k_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        assert x_train.ndim==1 and y_train.ndim==1,'dimension only can one'
        assert len(x_train)==len(y_train),'x_train length must equal with y_train'
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = (x_train-x_mean).dot(y_train-y_mean)
        d = (x_train-x_mean).dot(x_train-x_mean)
        self.k_ = num/d
        self.b_ = y_mean-self.k_*x_mean
        return self

    def predict(self,x_predict):
        assert x_predict.ndim==1,'x_predict must one dimension'
        y_predict = np.array([self.__predict(x) for x in x_predict])
        return y_predict

    def __predict(self,x):
        assert self.k_ and self.b_ is not None,'please fit() before predict()'
        y = self.k_*x+self.b_
        return y

    def mean_squared_error(self,x_test,y_test):
        y_predict = self.predict(x_test)
        mean_squared_error = np.sum((y_predict-y_test)**2)/len(y_test)
        return mean_squared_error

    def root_mean_squared_error(self,x_test,y_test):
        root_mean_squared_error = sqrt(self.mean_squared_error(x_test,y_test))
        return root_mean_squared_error

    def mean_absolute_error(self,x_test,y_test):
        y_predict = self.predict(x_test)
        mean_absolute_error = np.sum(np.absolute(y_test-y_predict))/len(y_test)
        return mean_absolute_error

    def r2_squared(self,x_test,y_test):
        mean_squared_error = self.mean_squared_error(x_test,y_test)
        r2_squared = 1-mean_squared_error/np.var(y_test)
        return r2_squared

# x = np.array([1.,2.,3.,4.,5.])
# y = np.array([1.,3.,2.,3.,5.])
# slr = SimpleLinearRegression()
# slr.fit(x,y)
# y_hat = slr.k_*x+slr.b_
# plt.scatter(x,y)
# plt.axis([0,6,0,6])
# plt.plot(x,y_hat,color='red')
# plt.show()

# m = 1000000
# x_data = np.random.random(size=m)
# y_data = x_data * 2 + 3 + np.random.normal(size=m)
# slr = SimpleLinearRegression()
# slr.fit(x_data,y_data)
# print(slr.k_)
# print(slr.b_)