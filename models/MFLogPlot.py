import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


#data =pd.read_csv('ft_of_cumulative_aa_freq_with_labels.csv')
#data =pd.read_csv('ft_of_integer_sequence_with_labels.csv')
#data =pd.read_csv('ft_of_kmer_freq_with_labels.csv')
#data =pd.read_csv('network_features_k_2_with_labels.csv')
#data =pd.read_csv('network_features_k_3_with_labels.csv')
#data =pd.read_csv('aa_composition_with_labels.csv')
data =pd.read_csv('kmer_frequency_with_labels.csv')
data=data.iloc[:,2:]
data.iloc[:,:-1]=StandardScaler().fit_transform(data.iloc[:,:-1])
data['label'].replace(to_replace = 1.0, value = 1, inplace = True)
data['label'].replace(to_replace = 2.0, value = 1, inplace = True)


train, test = sklearn.model_selection.train_test_split(data, train_size = 0.5,random_state=679)


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

a=(Weightsmat.iloc[max_index] != 0).index
#a = Weightsmat.columns.get_loc(a)
