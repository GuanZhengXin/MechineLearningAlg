import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# x = np.linspace(0,10,100)
# sinx = np.sin(x)
# cosx = np.cos(x)
# plt.plot(x,sinx,color='red',label='sin(x)',linestyle='-.')
# plt.plot(x,cosx,color='green',label='cos(x)',linestyle='--')
# plt.title('sin(x)--cos(x)')
# plt.legend()
# plt.show()

iris = datasets.load_iris()
# print(iris.keys())
# print(iris.data.shape)
#print(iris.feature_names)
#print(iris.data)

x = iris.data[:,:2] #只取萼片
y = iris.target #label
plt.scatter(x[y==0,0],x[y==0,1],color='red',marker='*',label='classifier-0')
plt.scatter(x[y==1,0],x[y==1,1],color='blue',marker='+',label='classifier-1')
plt.scatter(x[y==2,0],x[y==2,1],color='green',marker='x',label='classifier-2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.title('sepal')
plt.show()

x = iris.data[:,2:] #只取花瓣
y = iris.target #label
plt.scatter(x[y==0,0],x[y==0,1],color='red',marker='*',label='classifier-0')
plt.scatter(x[y==1,0],x[y==1,1],color='blue',marker='+',label='classifier-1')
plt.scatter(x[y==2,0],x[y==2,1],color='green',marker='x',label='classifier-2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.title('petal')
plt.show()
