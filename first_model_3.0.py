import arff
import csv
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
import pandas
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


# reader = csv.reader(open("training_features_3.2.csv", "r"), delimiter=",")
# data = np.array(list(reader))
# print(np.size(data, 1))
# print(data.shape)

#load data
path = "training_features_3.2.csv"
features = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81','f82','f83','f84','f85','f86','f87','f88','f89','f90','f91','f92','f93','f94','f95','f96','f97','f98','f99',
            'f100','f101','f102','f103','f104','f105','f106','f107','f108','f109','f110','f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f155','f156','f157','f158','f159','f160','f161','f162','f163','f164','f165','f166','f167','f168','f169','f170','f171','f172','f173','f174','f175','f176','f177','f178','f179','f180','f181','f182','f183','f184','f185','f186','f187','f188','f189','f190','f191','f192','f193','f194','f195','f196','f197','f198','f199',
            'f200','f201','f202','f203','f204','f205','f206','f207','f208','f209','f210','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233','f234','f235','f236','f237','f238','f239','f240','f241','f242','f243','f244','f245','f246','f247','f248','f249','f250','f251','f252','f253','f254','f255','f256','f257','f258','f259','f260','f261','f262','f263','f264','f265','f266','f267','f268','f269','f270','f271','f272','f273','f274','f275','f276','f277','f278','f279','f280','f281','f282','f283','f284','f285','f286','f287','f288','f289','f290','f291','f292','f293','f294','f295','f296','f297','f298','f299',
            'f300','f301','f302','f303','f304','f305','f306','f307','f308','f309','f310','f311','f312','f313','f314','f315','f316','f317','f318','f319','f320','f321','f322','f323','f324','f325','f326','f327','f328','f329','f330','f331','f332','f333','f334','f335','f336','f337','f338','f339','f340','f341','f342','f343','f344','f345','f346','f347','f348','f349','f350','f351','f352','f353','f354','f355','f356',	'f357','f358','f359','f360','f361','f362','f363','f364','f365','f366','f367','f368','f369','f370','f371','f372','f373','f374','f375','f376','f377','f378','f379','f380','f381','f382','f383','f384','f385','f386','f387','f388','f389','f390','f391','f392','f393','f394','f395','f396','f397','f398','f399',
            'f400','f401','f402','f403','f404','f405','f406','f407','f408','f409','f410','f411','f412','f413','f414','f415','f416','f417','f418','f419','f420','f421','f422','f423','f424','f425','f426','f427','f428','f429','f430','f431','f432','f433','f434','f435','f436','f437','f438','f439','f440','genre']

dataframe = pandas.read_csv(path, names=features);
# print(dataframe);
array = dataframe.values
# print(np.size(array,1))
X = array[:,0:384] #use all features except class label (type 1 or 0)
print(np.size(X,1))
print(X)
Y = array[:,384] #type 1 or type 0
print(np.size(Y))
print(Y);

# print("Univariate tests")
# # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification - need to work on this to see which one is best)
# # feature extraction
# test = SelectKBest(score_func=chi2, k=100) #choose the 100 best
# fit = test.fit(X, Y)
# # summarize scores
# np.set_printoptions(precision=3)
# #You can see the scores for each attribute and the 4 attributes chosen (those with the highest scores): plas, test, mass and age
# print(fit.scores_)
# features = fit.transform(X)
# # summarize selected features
# print(features)

print()
print("REF")
#Feature extraction with Recursive feature elimination
#feature extraction
model = LogisticRegression()
rfe = RFE(model, 100) #selecting top 4 features
fit = rfe.fit(X, Y)
print("Num features:", fit.n_features_)
print("Selected features:", fit.support_ ) #the features marked with 'True' are the selected features
print("Feature ranking:", fit.ranking_) #the '1' corresponds to the selected features
print()

print("PCA")
#Feature extraction with PCA
#https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
#I used this to understand what PCA does - it's very popular but I still need to get a hang of it (pls add on to it if anyone has experience with PCA)
# feature extraction
pca = PCA(n_components=100) #4 principal components
fit = pca.fit(X)
# summarize components
print("Explained Variance:", fit.explained_variance_ratio_)
print(fit.components_)
print()

print("Extra Trees Classifier")
#Feature Importance with Extra Trees Classifier - an 'importance score' is given for each attribute
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print("Feature importance scores", model.feature_importances_)
#
#     # result = np.array(x).astype("float")
# #    data = np.array(arff.load(open('librosa_mfcc_arff.arff'))['data'])
# #     names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f8']
# #     dataframe = pandas.read_csv("mfcc_librosa_updated_csv.csv", names=names);
#
# #     ## The combination of features, selected manually
# #     x = np.hstack((data[:,:208], data[:,332:360], data[:,314:322], data[:,324:325], data[:,328:329]))
# #
# #     #x = np.hstack((data[:,:208], data[:,332:360], data[:,324:325], data[:,328:329]))
# #     #x = np.hstack((data[:,:208], data[:,332:360]))
# #     #x = data[:,332:346]
# #     return x, data[:,-1]
# #
# # x, y = getData()
# # x = x.astype(float)
#
# # print(y);
# #y = y.astype(float)
#
# # Feature selection
# #lsvc = svm.SVC(probability=True)
# ##model = SelectFromModel(lsvc, prefit=True)
# ##x = model.transform(x)
# #
# #selector = RFECV(lsvc, step=1, cv=10)
# #selector = selector.fit(x, y)
# #x = selector.transform(x)
# #print(x.shape)
#
# #Feature selection
# #1 SelectKBest
#
#
#
# # clf_svm = svm.SVC(probability=True)
# # clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
# # clf_dtree = tree.DecisionTreeClassifier()
# # clf_logreg = LogisticRegression()
# # clf_gnb = GaussianNB()
# # clf_rfc = RandomForestClassifier()
# # clf_ada = AdaBoostClassifier()
# # clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
# #
# # eclf = VotingClassifier(estimators=[('svm', clf_svm), ('knn', clf_knn), ('log', clf_logreg)], voting='soft')
# #
# # # K-fold cross validation
# # n_splits = 10
# # kf = StratifiedKFold(n_splits=n_splits)
# # #kf = KFold(n_splits=n_splits, shuffle=True)
# # scores = []
# # c_mat = np.zeros((10,10))
# #
# #
# # for clf, label in zip([clf_svm, clf_knn, clf_dtree, clf_logreg, clf_gnb, clf_lda, clf_rfc, clf_ada, eclf], ['SVM', 'K-NN', 'Decision Tree', 'Log Regression', 'Naive Bayes', 'LDA', 'Random Forest', 'AdaBoost', 'Ensemble']):
# #
# #     for train_index, test_index in kf.split(x,y):
# #         x_train, x_test = x[train_index], x[test_index]
# #         y_train, y_test = y[train_index], y[test_index]
# #         clf.fit(x_train, y_train)
# #         y_pred = clf.predict(x_test)
# #         c_mat += confusion_matrix(y_test, y_pred)
# #         #scores.append(accuracy_score(y_test, y_pred)
# #
# #     ## Uncomment the below 2 lines if you
# #     ## want to print confusion matrix
# #     # print("Confusion matrix [%s]:" % label)
# #     # print(c_mat)
# #
# #     norm_c_mat = 100 * c_mat / c_mat.sum(axis=1)[:, np.newaxis]
# #     #print(norm_c_mat)
# #     scores = np.diagonal(norm_c_mat)
# #     # print("Accuracy: %0.2f%% [%s]" % (scores.mean(), label))
# #     #print(np.mean(scores))
# #
# #
# #     #scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
# #     #print(scores)
# #     #print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
