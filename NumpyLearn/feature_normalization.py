import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#特征数值归一化
#最值归一化 适合有边界的归一化
#x_scale = (x-x_min)/(x_max-x_min)
# data  = np.random.randint(0,100,size=(50,2))
# data = np.array(data,dtype=float)
# data[:,0] = (data[:,0]-np.min(data[:,0]))/(np.max(data[:,0])-np.min(data[:,0]))
# data[:,1] = (data[:,1]-np.min(data[:,1]))/(np.max(data[:,1])-np.min(data[:,1]))
# print(data)
# plt.scatter(data[:,0],data[:,1])
# plt.show()
# mean = np.mean(data[:,0])
# print(mean)
# std = np.std(data[:,0])
# print(std)

#均值方差归一化 适合无边界的归一化
#x_scale = (x-x_mean)/x_std
# data  = np.random.randint(0,100,size=(50,2))
# data = np.array(data,dtype=float)
# data[:,0] = (data[:,0]-np.mean(data[:,0]))/np.std(data[:,0])
# data[:,1] = (data[:,1]-np.mean(data[:,1]))/np.std(data[:,1])
# plt.scatter(data[:,0],data[:,1])
# plt.show()
# print(data)
# print(np.mean(data[:,0]))
# print(np.std(data[:,0]))

#训练测试数据集归一化
iris = datasets.load_iris()
X_train,X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=666)
standardScaler = StandardScaler()
standardScaler.fit(X_train)
mean = standardScaler.mean_
print(mean)
scale = standardScaler.scale_
print(scale)
X_train = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
score = knn.score(X_test_standard,y_test)
print(score)