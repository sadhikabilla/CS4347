import csv
import numpy as np
from sklearn import neighbors, svm
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def getData():
    data = csv.reader(open("training_features_3.2.csv", "r"), delimiter=",")
    data = np.array(list(data))
    x = np.hstack((data[:,:260], data[:,314:318], data[:,324:325], data[:,328:329], data[:,332:374], data[:,388:402]))
    return x.astype(float), data[:,-1]


clf_svm = svm.SVC(probability=True)
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_logreg = LogisticRegression()
clf_gnb = GaussianNB()
clf_lda = LinearDiscriminantAnalysis()

# Ensemble methods
clf_rfc = RandomForestClassifier()
clf_ada = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=10)
clf_voting = VotingClassifier(estimators=[('svm', clf_svm), ('log', clf_logreg), ('gnb', clf_gnb), ('lda', clf_lda), ('rfc', clf_rfc)], voting='soft')


x, y = getData()
classifiers = [clf_svm, clf_knn, clf_logreg, clf_gnb, clf_lda, clf_rfc, clf_ada, clf_voting]
labels = ['SVM', 'K-NN', 'Log Regression', 'Naive Bayes', 'LDA', 'Random Forest', 'AdaBoost', 'Voting classifier']

print("----- Accuracies -----")
for clf, label in zip(classifiers, labels):
    print(label, ":", round(cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean(), 3))
