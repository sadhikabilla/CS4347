import pandas
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names);
# print(dataframe);
array = dataframe.values
# print(np.size(array,1))
X = array[:,0:8] #use all features except class label (type 1 or 0)
print(np.size(X,1))
# print(X)
Y = array[:,8] #type 1 or type 0
print(np.size(Y))
# print(Y);

print("Univariate tests")
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification - need to work on this to see which one is best)
# feature extraction
test = SelectKBest(score_func=chi2, k=4) #choose the 4 best
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
#You can see the scores for each attribute and the 4 attributes chosen (those with the highest scores): plas, test, mass and age
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features)

print()
print("REF")
#Feature extraction with Recursive feature elimination
#feature extraction
model = LogisticRegression()
rfe = RFE(model, 4) #selecting top 4 features
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
pca = PCA(n_components=4) #4 principal components
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