import numpy as np

#arr = np.arange(10)
#arange() 生成0--n-1的二维数组 ,reshape(n,m) 改变维度
#arr = arr[1,1]
#ar = arr[:2,] #切片表达式
# ar =  arr[:2,1::2] #x:y ,n::m x-y行,开始为第n+1,间隔为m-1
# ar =  arr[0,:] #取第一行
# ar = arr[:,0] #取第一列 为一维数组
# ar = arr[:,0:1] #取第一列 是二维数组
# subArr = arr[:2,:3]
# subArr[0,0] = 100 #创建引用指向父矩阵  改变子矩阵会影响父矩阵
# subArr = arr[:2,:3].copy() # 改变子矩阵不会影响父矩阵
# subArr[0,0] = 100
#arr = arr.reshape(15)
# a1 = np.array([1,2,3])
# a2 = np.array([7,4,5,6])
# a3 = np.concatenate([a1,a2])  #连接两个list
# a1 = np.array([[1,2,3],[4,5,6]])
# a2 = np.array([[1,2,3],[7,8,9],[0,1,2]])
# a3 = np.concatenate([a1,a2]) #concatenate()只能处理相同维数
# arr = arr.reshape(1,-1)  #变成二维数组
# # a4 = np.concatenate([a3,arr],axis=1)

# a1 = np.array([[1,2,3],[4,5,6]])
# a2 = np.array([[1,2,3],[7,8,9],[0,1,2]])
# a3 = np.vstack([a1,a2])  #vstack 垂直合并

#分割
#arr = np.arange(16).reshape(4,4)
# print(arr)
# #results = np.split(arr,[2,3],axis=1) #分成几段 [n,m] n,m切割出索引
# #对应 也有vsplit 和 hsplit
# x1,x2 = np.hsplit(arr,[-1])
# x2 = x2[:,0]
# print(x1)
# print(x2)
#arr = arr.reshape(1,-1)
#arr = arr[0,:]
#print(arr)

#矩阵计算
# n = 10
# L = [i for i in range(10)]
# l = 2*L # L+L 连接
# print(l)
#
# l = np.arange(n)
# l = 2* l
# print(l)
# arr = np.arange(16).reshape(4,4)
# arr = arr//2
# print(arr)

#arr = np.arange(9).reshape(3,3)
# x = np.full((3,3),10)
# arr = arr * x # arr = arr.dot(x)  行*列
# print(arr)
#
# #arr = np.linalg.inv(arr) #求逆矩阵
# arr = np.linalg.pinv(arr) #求伪逆矩阵
# print(arr)

# x = np.arange(16)
# #乱序
# np.random.shuffle(x)
# print(x)
# random_x = np.sort(x)
# print(random_x)

#arg求得位置索引
# arr = np.random.normal(size=10000)
# max_arg = np.argmax(arr)
# min_arg = np.argmin(arr)
# print(max_arg)
# print(min_arg)

#二维
# arr = np.random.randint(20,size=(4,4))
# print(arr)
# arr.sort(axis=0)
# print(arr)
#np.partition(arr,kth=3) kth快速排序的节点


#FancyIndexing
# arr = np.arange(16)
# x = arr[3:9:2]
# index = [3,5,6,8]
# y = arr[index]
# print(x)
# print(y)

# arr = np.arange(16)
# arr = arr.reshape(4,-1)
# print(arr)
# row = np.array([0,2,3])
# col = np.array([0,1,3])
# x = arr[row,col]
# y = arr[0,col]
# z = arr[:2,col]
# print(x)
# print(y)
# print(z)

#函数
# x = np.arange(16)
# res = np.any(x==1)
# res = np.all(x==1)
# res = np.sum(x%2==0) #计算几个偶数count
# res = sum(x>2)
# res = np.count_nonzero(x)

# arr = np.arange(16).reshape(4,-1)
# x = arr[arr[:,3] % 3==0,3]
# y = arr[:,0]
# print(x)
# print(y)

distance = np.array([1.2,4,0.8,1.1,5.2,2,0.1])
min = np.argsort(distance)
max = np.argsort(-distance)
print(min)
print(max)
#print(distance)








