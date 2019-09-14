import numpy as np
from math import pow
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#自己写的KNN
class KnnClassifier():
    '''
    :param n_neighbors: n_neighbors
    :param weights: uniform: distance is equal weight , 'distance': Weight is the reciprocal of distance
    :param p: K nearest
    '''

    def __init__(self,n_neighbors=2,weights='uniform',p=2):
        assert n_neighbors>=1,'n_neighbors must >=1'
        assert weights=='uniform' or weights=='distance','undefined weights,weights must is uniform or distance'
        assert p >= 1, 'p must >=1'
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self,X_train,y_train):
        assert X_train.shape[0] >= 1, 'X_train must have value'
        assert X_train.shape[0] == y_train.shape[0],'dimension must be equal'
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        assert self.X_train is not None and self.y_train is not None,'before predict(),please fit()'
        for K in range(1,self.n_neighbors+1):
            if self.weights=='uniform':
                votes_list = [self.__uniform_predict(x_test) for x_test in X_test]
                votes = np.array(votes_list)
                return votes
            elif self.weights=='distance':
                predict_result = [self.__distance_predict(x_test) for x_test in X_test]
                votes = np.array(predict_result)
                return votes

    def __uniform_predict(self,x_test):
        distances = [pow(np.sum(abs(x_test-x_train)**2),1/self.p) for x_train in self.X_train]
        nearest = np.argsort(distances)
        y_predict = self.y_train[nearest[:self.n_neighbors]]
        votes_dic = Counter(y_predict)
        votes = votes_dic.most_common(1)[0][0]
        return votes

    #计算距离权重
    def __distance_predict(self,x_test):
        distances = np.array([pow(np.sum(abs(x_test - x_train) ** 2), 1 / self.p) for x_train in self.X_train],dtype=float)
        dis_nearest = np.argsort(distances)
        pre_distances = distances[dis_nearest[:self.n_neighbors]]
        weights = np.array([1/pre_distance for pre_distance in pre_distances])
        y_predict = self.y_train[dis_nearest[:self.n_neighbors]]
        max_y = max(y_predict)
        max_weight = 0.0
        predict_result = -1
        for i in range(0,max_y+1):
            sum_weight= 0.0
            i_nearest = np.argwhere(y_predict==i).flatten()
            for nearest in i_nearest:
                sum_weight += weights[nearest]
                if sum_weight > max_weight:
                    max_weight = sum_weight
                    predict_result = i
            print('{i}的权重和为{weight}'.format(i=i, weight=sum_weight))
        return predict_result


iris = datasets.load_iris()
X_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=666)
knn = KnnClassifier(n_neighbors=6,weights='distance')
knn.fit(X_train,y_train)
votes = knn.predict(x_test)
print(votes)
print(y_test)



