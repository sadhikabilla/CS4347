import arff
import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import KFold


def getData():
    data = np.array(arff.load(open('training_arff.arff'))['data'])
    # Here I'm keeping only the MFCC features
    # If you want all features, use data[:, :-1] (remove 21)
    return data[:,21:-1] , data[:,-1]

    
def trainSVM(x_train, x_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
#    print(y_test)
#    print(clf.predict(x_test))
    return clf.score(x_test, y_test)


def trainKNN(x_train, x_test, y_train, y_test):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)
    

x, y = getData()
score_knn = 0
score_svm = 0
n_neighbors = 5

# K-fold cross validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    score_knn += trainKNN(x_train, x_test, y_train, y_test)
    score_svm += trainSVM(x_train, x_test, y_train, y_test)


print("Accuracy for K-NN:", round(100*score_knn/n_splits, 2), "%")
print("Accuracy for SVM:", round(100*score_svm/n_splits, 2), "%")
