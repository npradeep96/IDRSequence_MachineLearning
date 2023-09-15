import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn

#data =pd.read_pickle('normData.pkl')
data = pd.read_pickle('preprocessed178.pkl')
standard = 1

train, test = sklearn.model_selection.train_test_split(data, train_size = 0.5,random_state=679)

X_train=train.iloc[:,:-1]
X_test=test.iloc[:,:-1]

# data we are importing isn't already standardized
if standard== 0:
     #standardize the data
     sc = StandardScaler()
     scaler = sc.fit(X_train) # fit on training set only
     # apply transform to both training and test set
     X_train = scaler.transform(X_train)
     X_test = scaler.transform(X_test)

y_train=train.iloc[:,-1]
y_test=test.iloc[:,-1]

#implement mrmr
#mrmr=pd.read_pickle('mrmr_10to60_alpha0.5_MID_ThursDec8.pkl')
#column_names=mrmr[0] #edit index for different mrmr
#column_names.append('label')
#colInd = data.columns.get_indexer(column_names) # numerical index of which columns to keep mrmr
#X_test=X_test[:,colInd]
#X_train=X_train[:,colInd]


# Train the neural network
# max_iter is max number of iterations. for stochastic solves this is the 
# number of epoches not the number of gradient steps
# alpha is the strength of the L2 regularization term
# activation is the activation function for the hidden layers
# solver is the algorithm for weight optimization over the nodes 
# hidden layer size = (i,j,k,....) i is number of nodes in first hidden layer, j is number of nodes in second hidden layer, etc....
model = MLPClassifier(hidden_layer_sizes = (10) , max_iter=1000,alpha=1e-3, activation = 'relu', solver = 'adam', tol = 1e-4)
model.fit(X_train, y_train)


# Make predictions on the test set
predictions = model.predict(X_test)
predictions_train=model.predict(X_train)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
accuracy_train = accuracy_score(y_train, predictions_train)

# balanced accurary score
accBal_test = sklearn.metrics.balanced_accuracy_score(y_test,predictions)
accBal_train = sklearn.metrics.balanced_accuracy_score(y_train,predictions_train)

# Print the accuracy
print('Accuracy:', accuracy)
print('Accuracy train:', accuracy_train)

# Print the balanced accurary
print('Balanced accurary:', accBal_test)
print('Balanced accurcy train:', accBal_train)

plt.plot(model.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
