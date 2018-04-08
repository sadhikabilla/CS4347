import arff
import numpy as np
from sklearn import neighbors, svm, tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def getData():
    data = np.array(arff.load(open('basic_arff.arff'))['data'])
    ## Here I'm keeping only the MFCC features
    ## If you want all features, use data[:, :-1] (remove 21)
    # return data[:,21:-1] , data[:,-1]
    return data[:,21:-1] , data[:,-1]
  
x, y = getData()
x = x.astype(float)
#y = y.astype(float)

clf_svm = svm.SVC(probability=True)
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_dtree = tree.DecisionTreeClassifier()
clf_logreg = LogisticRegression()
clf_gnb = GaussianNB() 
clf_rfc = RandomForestClassifier()
clf_ada = AdaBoostClassifier()

eclf = VotingClassifier(estimators=[('svm', clf_svm), ('knn', clf_knn), ('log', clf_logreg)], voting='hard')

# K-fold cross validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)
scores = []
c_mat = np.zeros((10,10))


for clf, label in zip([clf_svm, clf_knn, clf_dtree, clf_logreg, clf_gnb, clf_rfc, clf_ada, eclf], ['SVM', 'K-NN', 'Decision Tree', 'Log Regression', 'Naive Bayes', 'Random Forest', 'AdaBoost', 'Ensemble']):
    
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        c_mat += confusion_matrix(y_test, y_pred)
        #scores.append(accuracy_score(y_test, y_pred)

    ## Uncomment the below 2 lines if you
    ## want to print confusion matrix   
    # print("Confusion matrix [%s]:" % label)
    # print(c_mat)
    
    norm_c_mat = 100 * c_mat / c_mat.sum(axis=1)[:, np.newaxis]
    #print(norm_c_mat)
    scores = np.diagonal(norm_c_mat)
    print("Accuracy: %0.2f%% [%s]" % (scores.mean(), label))
    #print(np.mean(scores))
    
    
    #scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
    #print(scores)
    #print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
