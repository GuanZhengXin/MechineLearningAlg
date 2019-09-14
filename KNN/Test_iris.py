from sklearn import datasets
from KNN.KNNClassifier import KNNClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV

def train_test_split(X_train,Y_train,test_rate=0.2):
    assert X_train.shape[0] == Y_train.shape[0],'train shape is not equal label shape'
    assert 0.0<=test_rate<=1.0,'random_rate must in 0-1'
    shuffle_indeics = np.random.permutation(len(X_train))
    test_number = int(test_rate * len(X_train))
    X_test_train = X_train[shuffle_indeics[:test_number]]
    X_train = X_train[shuffle_indeics[test_number:]]

    Y_test_train = Y_train[shuffle_indeics[:test_number]]
    Y_train = Y_train[shuffle_indeics[test_number:]]
    return  X_train,Y_train,X_test_train,Y_test_train

def GetScore(Y_test_label,Y_predict_label):
    assert Y_test_label.shape[0]==Y_predict_label.shape[0],'test train data axis must be equal to test train label'
    assert Y_test_label.shape[0]>=1,'label is not exist'
    result = np.sum(Y_test_label==Y_predict_label)/Y_test_label.shape
    return result

def GetBestScoreAndKBySelf(start_K,end_K,X_train,Y_train,X_test_train,Y_test_train):
    assert start_K>=1,'start_K must >=1'
    assert start_K <= end_K,'start_K must <=end_K'
    best_score = 0.0
    best_K = -1
    for K in range(start_K,end_K+1):
        knn = KNNClassifier(K=K)
        knn.fit(X_train, Y_train)
        y_predict_label = knn.predict(X_test_train)
        score = GetScore(Y_test_train, y_predict_label)
        if score > best_score:
            best_score = score
            best_K = K
    return best_score,best_K

def GetBestScoreAndK(start_K,end_K,X_train,Y_train,X_test_train,Y_test_train):
    assert start_K >= 1, 'start_K must >=1'
    assert start_K <= end_K, 'start_K must <=end_K'
    best_score = 0.0
    best_K = -1
    best_p = -1
    for p in range(1,10):
        for K in range(start_K, end_K + 1):
            knn = KNeighborsClassifier(n_neighbors=K,p=p)
            knn.fit(X_train, Y_train)
            y_predict_label = knn.predict(X_test_train)
            score = GetScore(Y_test_train, y_predict_label)
            if score > best_score:
                best_score = score
                best_K = K
                best_p = p
    return best_score, best_K,best_p

# iris = datasets.load_iris()
# X_train,Y_train,X_test_train,Y_test_train = train_test_split(iris.data,iris.target)
# print('测试标签为{y_test_train}'.format(y_test_train=Y_test_train))
# knn = KNNClassifier(K=4) #K为超参数
# knn.fit(X_train,Y_train)
# y_predict_label = knn.predict(X_test_train)
# print('预测结果为{result}'.format(result = y_predict_label))
# accuracy = GetScore(Y_test_train,y_predict_label)
# print('模型准确率为{accuracy}'.format(accuracy=accuracy))

#超参数选择 不同K值得分数
# iris = datasets.load_iris()
# X_train,Y_train,X_test_train,Y_test_train = train_test_split(iris.data,iris.target)
# print('测试标签为{y_test_train}'.format(y_test_train=Y_test_train))
# best_score,best_K= GetBestScoreAndKBySelf(1,1,X_train,Y_train,X_test_train,Y_test_train)
# print('最好的分数为{score},对应的K为{K}'.format(score=best_score,K=best_K))

#p代表另一个超参数
# iris = datasets.load_iris()
# X_train,Y_train,X_test_train,Y_test_train = train_test_split(iris.data,iris.target)
# print('测试标签为{y_test_train}'.format(y_test_train=Y_test_train))
# best_score,best_K,best_p = GetBestScoreAndK(1,11,X_train,Y_train,X_test_train,Y_test_train)
# print('最好的分数为{score},对应的K为{K},p为{p}'.format(score=best_score,K=best_K,p=best_p))

#网格搜索 SearchGrid
iris = datasets.load_iris()
X_train,Y_train,X_test_train,Y_test_train = train_test_split(iris.data,iris.target)
print('测试标签为{y_test_train}'.format(y_test_train=Y_test_train))
params_grid = [
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn,params_grid,n_jobs=1,verbose=2) #n_jobs分配几个内核 ,verbose在gridsearch时输出内容
grid_search.fit(X_train,Y_train)
best_estimator = grid_search.best_estimator_
best_score = grid_search.best_score_
best_params = grid_search.best_params_
score = grid_search.score(X_test_train,Y_test_train)
print(score)