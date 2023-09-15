import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data =pd.read_pickle('normData.pkl')
#data=data.iloc[:,[10, 22, 71, 162, 172, 239, 260, 309, 345, 372,383]] #mrmr 10 features +label
train, test = sklearn.model_selection.train_test_split(data, train_size = 0.5,random_state=679)
X_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]
X_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]

CMat=np.logspace(-3, 3, num=13)


trainscore = []
testscore = []
coeffL2=[];
for x in CMat:
    clf=LogisticRegression(penalty='l2',C=x,max_iter=10000)
    clf.fit(X_train, y_train)
    trainscore.append(clf.score(X_train, y_train))
    testscore.append(clf.score(X_test, y_test))
    coeffL2.append(clf.coef_)

    
trainscorel1 = []
testscorel1 = []
coeffL1=[]
for x in CMat:
    clf=LogisticRegression(penalty='l1',C=x,max_iter=10000,solver='liblinear')
    clf.fit(X_train, y_train)
    trainscorel1.append(clf.score(X_train, y_train))
    testscorel1.append(clf.score(X_test, y_test))
    coeffL1.append(clf.coef_)

logc=np.linspace(-3, 3, num=13)
# Create the plot and set the label for each data series
plt.plot(logc, trainscore, label="l2train")
plt.plot(logc, testscore, label="l2test")
plt.plot(logc, trainscorel1, label="l1train")
plt.plot(logc, testscorel1, label="l1test")
plt.xlabel("log_10(C)")
plt.ylabel("Accuracy")

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()  
    