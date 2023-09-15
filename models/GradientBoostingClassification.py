import numpy as np 
import sklearn
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

#data =pd.read_pickle('normData.pkl')
data = pd.read_pickle('preprocessed470.pkl')

shp = np.shape(data)
N =  int(shp[0]) # number of data points
nF = int(shp[1]) # number of features

train, test = sklearn.model_selection.train_test_split(data, train_size = 0.5,random_state=679)

X_train=train.iloc[:,:-1]
X_test=test.iloc[:,:-1]
y_train=train.iloc[:,-1]
y_test=test.iloc[:,-1]


#standardize the data
sc = StandardScaler()
scaler = sc.fit(X_train) # fit on training set only
# apply transform to both training and test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# evaluate the model
#model = GradientBoostingClassifier()
#clf = XGBClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
clf = GradientBoostingClassifier(n_estimators=150, learning_rate=1.0, min_weight_fraction_leaf = 0.1, max_depth=3, random_state=0, max_features = 50).fit(X_train, y_train)

#print(clf.score(X_test,y_test))

balAc = sklearn.metrics.balanced_accuracy_score(y_test,clf.predict(X_test))
print(balAc)


#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
## fit the model on the whole dataset
#model = GradientBoostingClassifier()
#model.fit(X, y)


