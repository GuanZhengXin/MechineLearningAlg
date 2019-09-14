from sklearn import datasets
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier

#kNN算法 手动
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
plt.scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],color='red',label='X-Label')
plt.scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],color='blue',label='Y-Label')
plt.xlabel('X-Feature')
plt.ylabel('Y-Feature')
#新样本 比如取4NN
K = 4
new_sample = np.array([5.0936,3.3657])
plt.scatter(new_sample[0],new_sample[1],color='green')
plt.show()
distance =[sqrt(np.sum((x_train-new_sample)**2)) for x_train in X_train]
distance_index = np.argsort(distance)
k_distance_index = distance_index[:K]
label_votes = Y_train[k_distance_index]
print(label_votes)
votes = Counter(label_votes) #字典计数
print(votes)
result = votes.most_common(1)
result = result[0][0]
print('新目标x={x_lable},y={y_label}得病的大概结果是{result}'.format(x_lable=new_sample[0],y_label=new_sample[1],result=result))


#kNN算法 sklearn的kNN算法
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
new_sample = np.array([5.0936,3.3657]).reshape(1,-1) #传矩阵
knn_classifer = KNeighborsClassifier(n_neighbors=4)
knn_classifer.fit(X_train,Y_train)
predict = knn_classifer.predict(new_sample)
print(predict)

