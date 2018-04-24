import numpy as np
import pandas as pd
import pickle
import os.path
from xgboost import XGBClassifier
from sklearn import neighbors, svm, tree
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

#File path for training data
path = "features_train.csv"

#440 features in total
features = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81','f82','f83','f84','f85','f86','f87','f88','f89','f90','f91','f92','f93','f94','f95','f96','f97','f98','f99',
            'f100','f101','f102','f103','f104','f105','f106','f107','f108','f109','f110','f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f155','f156','f157','f158','f159','f160','f161','f162','f163','f164','f165','f166','f167','f168','f169','f170','f171','f172','f173','f174','f175','f176','f177','f178','f179','f180','f181','f182','f183','f184','f185','f186','f187','f188','f189','f190','f191','f192','f193','f194','f195','f196','f197','f198','f199',
            'f200','f201','f202','f203','f204','f205','f206','f207','f208','f209','f210','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233','f234','f235','f236','f237','f238','f239','f240','f241','f242','f243','f244','f245','f246','f247','f248','f249','f250','f251','f252','f253','f254','f255','f256','f257','f258','f259','f260','f261','f262','f263','f264','f265','f266','f267','f268','f269','f270','f271','f272','f273','f274','f275','f276','f277','f278','f279','f280','f281','f282','f283','f284','f285','f286','f287','f288','f289','f290','f291','f292','f293','f294','f295','f296','f297','f298','f299',
            'f300','f301','f302','f303','f304','f305','f306','f307','f308','f309','f310','f311','f312','f313','f314','f315','f316','f317','f318','f319','f320','f321','f322','f323','f324','f325','f326','f327','f328','f329','f330','f331','f332','f333','f334','f335','f336','f337','f338','f339','f340','f341','f342','f343','f344','f345','f346','f347','f348','f349','f350','f351','f352','f353','f354','f355','f356','f357','f358','f359','f360','f361','f362','f363','f364','f365','f366','f367','f368','f369','f370','f371','f372','f373','f374','f375','f376','f377','f378','f379','f380','f381','f382','f383','f384','f385','f386','f387','f388','f389','f390','f391','f392','f393','f394','f395','f396','f397','f398','f399',
            'f400','f401','f402','f403','f404','f405','f406','f407','f408','f409','f410','f411','f412','f413','f414','f415','f416','f417','f418','f419','f420','f421','f422','f423','f424','f425','f426','f427','f428','f429','f430','f431','f432','f433','f434','f435','f436','f437','f438','f439','f440','f441','f442','f443','f444','f445','f446','f447','f448','f449','f450','f451','f452','f453','f454','f455','f456','f457','f458','f459','f460','f461','f462','f463','f464','f465','f466','f467','f468','f469','f470', 'genre']

dataframe = pd.read_csv(path, names=features)
array = dataframe.values
# Array with the 470 features
X = array[:, 0:470]
# Array with genre name
Y = array[:, 470]


# Feature selection with Recursive feature elimination
def rfe(nb_features):
    print("RFE with", nb_features, "features")
    model = LogisticRegression()
    rfe = RFE(model, nb_features)  # selecting top features
    fit_data_dmp = 'fit_data.dmp'
    if os.path.isfile(fit_data_dmp):
        with open(fit_data_dmp,'rb') as f:
            fit_data = pickle.load(f)
    else:
        fit_data = rfe.fit(X,Y)
        with open(fit_data_dmp, 'wb') as f:
            pickle.dump(fit_data,f) #saves the RFE model to the disk
    # print("Num features:", rfe.fit(X, Y).n_features_)
    # print("Selected features:", rfe.fit(X, Y).support_)  # the features marked with 'True' are the selected features
    # print("Feature ranking:", rfe.fit(X, Y).ranking_)  # the '1' corresponds to the selected features
    filter_condition = np.hstack((fit_data.ranking_ == 1, [False]))
    #rfe.fit(X, Y)
    filtered_matrix = array[:, filter_condition]
    # print(filtered_matrix)
    return filtered_matrix.astype(float), array[:, -1], filter_condition[:-1]

#selecting the top 150 features
x, y, selection_filter = rfe(150)


clf_svm = svm.SVC(probability=True)
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_dtree = tree.DecisionTreeClassifier()
clf_logreg = LogisticRegression()
clf_gnb = GaussianNB()
clf_lda = LinearDiscriminantAnalysis()
clf_rfc = RandomForestClassifier()
clf_ada = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=10)
clf_xgb = XGBClassifier() # To tune the parameters: http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn #to add
clf_voting = VotingClassifier(estimators=[('log', clf_logreg), ('gnb', clf_gnb), ('lda', clf_lda), ('rfc', clf_rfc), ('xgb', clf_xgb)], weights=[3, 1, 3, 1, 3], voting='soft')
#clf_voting = VotingClassifier(estimators=[('svm', clf_svm), ('log', clf_logreg), ('gnb', clf_gnb), ('lda', clf_lda), ('rfc', clf_rfc)], voting='soft') #to delete
clf_voting2 = VotingClassifier(estimators=[('log', clf_logreg), ('lda', clf_lda), ('xgb', clf_xgb)], weights=[2, 1, 1], voting='soft')
clf_voting3 = VotingClassifier(estimators=[('svm', clf_svm), ('log', clf_logreg), ('lda', clf_lda), ('xgb', clf_xgb)], weights=[1, 2, 1, 1], voting='soft') #BEST
clf_voting4 = VotingClassifier(estimators=[('svm', clf_svm), ('log', clf_logreg), ('lda', clf_lda), ('xgb', clf_xgb), ('knn', clf_knn), ('gnb', clf_gnb)], weights=[2, 3, 2, 2, 1, 1], voting='soft')

#classifiers = [clf_svm, clf_knn, clf_dtree, clf_logreg, clf_gnb, clf_lda, clf_xgb, clf_voting, clf_voting2] #to comment
#labels = ['SVM', 'K-NN', 'Decision Tree', 'Logistic Regression', 'Gaussian Naive Bayes', 'LDA', 'XGBoost', 'Voting', 'Voting2'] #to comment
#classifiers = [clf_voting, clf_voting2, clf_voting3, clf_voting4]
#labels = ['Voting', 'Voting2', 'Voting3', 'Voting4']
classifiers = [clf_voting3]
labels = ['Voting3']


print("----- Accuracies -----")
#for clf, label in zip(classifiers, labels):
#     print(label, ": %0.2f%%" % (cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()*100))

# K-fold cross validation
n_splits = 10
kf = StratifiedKFold(n_splits=n_splits)
scores = []
    
c_mat = np.zeros((10,10))

for train_index, test_index in kf.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_voting3.fit(x_train, y_train)
    y_pred = clf_voting3.predict(x_test)
    c_mat += confusion_matrix(y_test, y_pred)
    #scores.append(accuracy_score(y_test, y_pred)
  
print("Confusion matrix of Voting3")
print(c_mat)
print("Normalised confusion matrix of Voting3")
norm_c_mat = c_mat / c_mat.sum(axis=1)[:, np.newaxis]
print(norm_c_mat)
print("Accuracy of Voting3")
scores = np.diagonal(norm_c_mat)
print(scores.mean())

# For predicting test dataset
fout = open("prediction.txt", "w")

dataframe = pd.read_csv('features_test.csv', header=None)
x_test = dataframe.values
selected_x_test = x_test[:, selection_filter]
selected_x_test = selected_x_test.astype(float)

clf_voting3.fit(x, y)
y_pred = clf_voting3.predict(selected_x_test)

for i in range(len(y_pred)):
    print(y_pred[i], file=fout)
fout.close()