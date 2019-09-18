__author__ = 'Guan'
import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier():
    def __init__(self,K):
        assert K>=1,'K应该大于等于1'
        self.K = K
        self.X_train = None
        self.Y_train = None

    def fit(self,X_train,Y_train):
        assert X_train.shape[0] == Y_train.shape[0],'x，y训练集应纬度相同'
        assert self.K<= X_train.shape[0],'K值个数应小于数据维度'
        self.X_train = X_train
        self.Y_train = Y_train
        return self

    def predict(self,X_predicts):
        assert self.X_train is not None and self.Y_train is not None,'必须先拟合模型,请使用fit()'
        assert X_predicts.shape[1] == self.X_train.shape[1],'数据维度不相同'
        predicts = [self.pre(x) for x in X_predicts]
        predicts = np.array(predicts)
        return predicts

    def pre(self,x):
        distances = [sqrt(np.sum((x_train-x)**2)) for x_train in self.X_train]
        distance_indeics = np.argsort(distances)
        topK = [self.Y_train[i] for i in distance_indeics[:self.K]]
        votes = Counter(topK)
        result = votes.most_common(1)
        return  result[0][0]

    def __repr__(self):
        return 'KNN(K=%d)' % self.K

#实现自己的KNNClassifier算法
x = ([[3.3935,2.3312],
     [3.1100,1.7815],
     [1.3438,3.3683],
     [3.5822,4.6791],
     [2.2803,2.8669],
     [7.4234,4.6965],
     [5.7450,3.5339],
     [9.1721,2.5110],
     [7.7927,3.4240],
     [7.9398,0.7916]])
y = ([0,0,0,0,0,1,1,1,1,1])
X_train = np.array(x)
Y_train = np.array(y)
new_sample = np.array([[3.4537,6.8845],[5.0936,3.3657]])
knn = KNNClassifier(K=4)
knn.fit(X_train,Y_train)
result = knn.predict(new_sample)
print(result)