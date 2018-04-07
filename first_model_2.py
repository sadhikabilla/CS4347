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


def getData():
    data = np.array(arff.load(open('basic_arff.arff'))['data'])
    # Here I'm keeping only the MFCC features
    # If you want all features, use data[:, :-1] (remove 21)
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


eclf = VotingClassifier(estimators=[('svm', clf_svm), ('log', clf_logreg)], voting='soft')

for clf, label in zip([clf_svm, clf_knn, clf_dtree, clf_logreg, clf_gnb, clf_rfc, clf_ada, eclf], ['SVM', 'K-NN', 'Decision Tree', 'Log Regression', 'Naive Bayes', 'Random Forest', 'AdaBoost', 'Ensemble']):
    scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
