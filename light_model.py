import csv
import numpy as np
from xgboost import XGBClassifier
from sklearn import neighbors, svm, tree
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def getData():
    data = csv.reader(open("features_train.csv", "r"), delimiter=",")
    data = np.array(list(data))
    #x = np.hstack((data[:,:156], data[:,208:260], data[:,312:314], data[:,318:320], data[:,322:324], data[:,326:328], data[:,338:380], data[:,394:408]))
    x = np.hstack((data[:,:104], data[:,104:156], data[:,208:260], data[:,312:314], data[:,314:316], data[:,318:320], data[:,322:324], data[:,326:328], data[:,330:334], data[:,334:336], data[:,338:366], data[:,366:380], data[:,394:408]))
    return x.astype(float), data[:,-1]


clf_svm = svm.SVC(probability=True)
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_dtree = tree.DecisionTreeClassifier()
clf_logreg = LogisticRegression()
clf_gnb = GaussianNB()
clf_lda = LinearDiscriminantAnalysis()

# Ensemble methods
clf_rfc = RandomForestClassifier()
clf_ada = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=10)
clf_xgb = XGBClassifier() # To tune the parameters: http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
clf_voting = VotingClassifier(estimators=[('log', clf_logreg), ('gnb', clf_gnb), ('lda', clf_lda), ('rfc', clf_rfc), ('xgb', clf_xgb)], weights=[3, 1, 3, 1, 3], voting='soft')


x, y = getData()
print("Feature vector dimension: ", len(x[0]))
#classifiers = [clf_svm, clf_knn, clf_dtree, clf_logreg, clf_gnb, clf_lda, clf_rfc, clf_ada]
#labels = ['SVM', 'K-NN', 'Decision Tree', 'Logistic Regression', 'Gaussian Naive Bayes', 'LDA', 'Random forest', 'AdaBoost']
classifiers = [clf_voting]
labels = ['Voting classifier']

print("----- Accuracies -----")
for clf, label in zip(classifiers, labels):
    print(label, ": %0.2f%%" % (cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()*100))
