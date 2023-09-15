import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


data =pd.read_pickle('normData.pkl')
data.iloc[:,:-1]=StandardScaler().fit_transform(data.iloc[:,:-1])

train, test = sklearn.model_selection.train_test_split(data, train_size = 0.5,random_state=679)

#implement mrmr
mrmr=pd.read_pickle('mrmr_10to60_alpha0.5_MID_ThursDec8.pkl')
column_names=mrmr[5] #edit index for different mrmr
column_names.append('label')
test=test.loc[:,column_names]
train=train.loc[:,column_names]


X_train=train.iloc[:,:-1]
X_test=test.iloc[:,:-1]
y_train=train.iloc[:,-1]
y_test=test.iloc[:,-1]


CMat=np.logspace(-3, 3, num=13)
trainscore = []
testscore = []
coeffL2=[];

# do L2 and iterate through param C
for x in CMat:
    clf=LogisticRegression(penalty='l2',C=x,max_iter=10000,class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred_test=clf.predict(X_test)
    y_pred_train=clf.predict(X_train)
    trainscore.append(sklearn.metrics.balanced_accuracy_score(y_train,y_pred_train))
    testscore.append(sklearn.metrics.balanced_accuracy_score(y_test,y_pred_test))
    coeffL2.append(clf.coef_)

    
trainscorel1 = []
testscorel1 = []
coeffL1=[];
#do L1 and iterate through param C
for x in CMat:
    clf=LogisticRegression(penalty='l1',C=x,max_iter=10000,solver='liblinear',class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred_test=clf.predict(X_test)
    y_pred_train=clf.predict(X_train)
    trainscorel1.append(sklearn.metrics.balanced_accuracy_score(y_train,y_pred_train))
    testscorel1.append(sklearn.metrics.balanced_accuracy_score(y_test,y_pred_test))
    coeffL1.append(clf.coef_)

#save coeffs of L1
Weightsmat=pd.DataFrame(data=np.squeeze(coeffL1))
Dummy=test.iloc[:,:-1]
Weightsmat.columns = Dummy.columns


logc=np.linspace(-3, 3, num=13)
# Create the plot and set the label for each data series
plt.plot(logc, trainscore, label="L2-train")
plt.plot(logc, testscore, label="L2-test")
plt.plot(logc, trainscorel1, label="L1-train")
plt.plot(logc, testscorel1, label="L1-test")
plt.xlabel("$log_{10}(C)$")
plt.ylabel("Balanced Accuracy")

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()  

print(max(testscorel1))
max_index = testscorel1.index(max(testscorel1))
num_nonzero = (Weightsmat.iloc[max_index] != 0).sum()
print(num_nonzero)


