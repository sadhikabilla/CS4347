import arff
import csv
import numpy as np
from sklearn import neighbors, svm, tree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFECV

def getData():
    reader = csv.reader(open("mfcc_librosa_updated_csv.csv", "r"), delimiter=",")
    data = np.array(list(reader))
#    result = numpy.array(x).astype("float")
#    data = np.array(arff.load(open('librosa_mfcc_arff.arff'))['data'])

    ## The combination of features, selected manually
    x = np.hstack((data[:,:208], data[:,332:360], data[:,314:322], data[:,324:325], data[:,328:329]))    
    
    #x = np.hstack((data[:,:208], data[:,332:360], data[:,324:325], data[:,328:329]))     
    #x = np.hstack((data[:,:208], data[:,332:360]))
    #x = data[:,332:346]  
    return x, data[:,-1]
  
x, y = getData()
x = x.astype(float)
#y = y.astype(float)

# Feature selection
#lsvc = svm.SVC(probability=True)
##model = SelectFromModel(lsvc, prefit=True)
##x = model.transform(x)
#
#selector = RFECV(lsvc, step=1, cv=10)
#selector = selector.fit(x, y)
#x = selector.transform(x)
#print(x.shape)


clf_svm = svm.SVC(probability=True)
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_dtree = tree.DecisionTreeClassifier()
clf_logreg = LogisticRegression()
clf_gnb = GaussianNB() 
clf_rfc = RandomForestClassifier()
clf_ada = AdaBoostClassifier()
clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)

eclf = VotingClassifier(estimators=[('svm', clf_svm), ('knn', clf_knn), ('log', clf_logreg)], voting='soft')

# K-fold cross validation
n_splits = 10
kf = StratifiedKFold(n_splits=n_splits)
#kf = KFold(n_splits=n_splits, shuffle=True)
scores = []
c_mat = np.zeros((10,10))


for clf, label in zip([clf_svm, clf_knn, clf_dtree, clf_logreg, clf_gnb, clf_lda, clf_rfc, clf_ada, eclf], ['SVM', 'K-NN', 'Decision Tree', 'Log Regression', 'Naive Bayes', 'LDA', 'Random Forest', 'AdaBoost', 'Ensemble']):
    
    for train_index, test_index in kf.split(x,y):
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
