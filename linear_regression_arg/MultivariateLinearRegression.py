from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from math import  sqrt

#多元线性回归
#theta = inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

class LinearRegression():

    def __init__(self):
        self.coef_= None
        self.interception_ = None
        self.theta_ = None

    def fit_normal(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0],'X_train size must equal to y_train'
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        self.theta_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self.theta_[1:]
        self.interception_ = self.coef_[0]
        return  self

    def fit_gradient_descent(self,X_train,y_train,eta=0.1,epsilon=1e-8):
        assert X_train.shape[0] == y_train.shape[0],'shape must have be equal'

        def J(theta,X_b,y):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(X_b)
            except:
                return float('inf')

        def Dj(theta,X_b,y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta)-y)/len(X_b)
            for i in range(1,len(theta)):
                res[i] = (X_b.dot(theta)-y).dot(X_b[:,i])

            return  res * 2 / len(X_b)

        def gradient_descent(initial_theta,X_b,y,counter=1e4):
            theta =initial_theta
            i=0
            while i<counter:
                gradient = Dj(theta,X_b,y)
                last_theta = theta
                theta = theta - eta * gradient

                if(abs(J(theta,X_b,y)-J(last_theta,X_b,y))<=epsilon):
                    break
                i+=1
            return theta

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1]) #特征数个数 thea个数
        self.theta_ = gradient_descent(initial_theta,X_b,y_train)
        self.interception_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self


    def predict(self,X_test):
        assert self.coef_  is not  None and self.interception_ is not  None,'must fit() please'
        assert len(self.coef_) == X_test.shape[1] , 'size is not equal'
        X_b = np.hstack([np.ones((len(X_test), 1)), X_test])
        y_predict = X_b.dot(self.theta_)
        return y_predict

    def score(self,X_test,y_test):
        r2_squared = self.r2_squared(X_test,y_test)
        return r2_squared

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

# boston = datasets.load_boston()
# x = boston.data[boston.target<50]
# y = boston.target[boston.target<50]
# X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=666)
# LiR = LinearRegression()
# LiR.fit_normal(X_train,y_train)
# print(LiR.coef_)
# r2_squared = LiR.score(X_test,y_test)
# print(r2_squared)

x = 2*np.random.random(size=100)
y = x*3. + 4. + np.random.normal(size=100)
x = x.reshape(-1,1)
print(x.shape)
print(y.shape)
lg = LinearRegression()
lg.fit_gradient_descent(x,y)
print(lg.interception_,lg.coef_)
