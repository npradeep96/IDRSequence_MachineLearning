import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# reference for PCA in sklearn
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

data =pd.read_pickle('normData.pkl')

train, test = sklearn.model_selection.train_test_split(data, train_size = 0.5,random_state=679)

# Make an instance of the PCA Model
pca = PCA(.95) # keep number of features needed to explain 95% of variance

	# standardize the data
X_train=train.iloc[:,:-1]
X_test=test.iloc[:,:-1]
scaler = StandardScaler()
scaler.fit(X_train) # fit on training set only
# apply transform to both training and test set 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train=train.iloc[:,-1]
y_test=test.iloc[:,-1]

# fit PCA on training set ONLY
pca.fit(X_train)

# apply mapping to both training set and test set
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
fNum = np.shape(X_train)
print('Number of features kept for PCA')
print(fNum[1])
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
coeffdata=np.squeeze(coeffL1)
#print(np.shape(coeffdata))
#quit()
#Weightsmat=pd.DataFrame(data=np.squeeze(coeffL1))
#Dummy=test.iloc[:,:-1]
#Weightsmat.columns = Dummy.columns

# print out the max accuracy of testing
maxL2 = np.max(testscore)
ind2 = np.argmax(testscore)
maxL1 = np.max(testscorel1)
ind1 = np.argmax(testscorel1)

print('max accurary L1')
print(maxL1)
print('number of zeros in weight matrix')
nZeroWeight = np.count_nonzero(coeffdata[ind1,:]==0)
print(nZeroWeight)
print('max accuracy L2')
print(maxL2)
quit()


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
