import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sklearn
import numpy as np

data =pd.read_pickle('normData.pkl')
#data=data.iloc[:,[10, 22, 71, 162, 172, 239, 260, 309, 345, 372,383]] #mrmr 10 features +label
train, test = sklearn.model_selection.train_test_split(data, train_size = 0.7,random_state=679)
X_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]
X_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]



CMat=np.logspace(-3, 3, num=13)

trainscore = []
testscore = []
coeffL2=[];
for x in CMat:
    # Train the SVM model
    model = SVC()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    predictions_train=model.predict(X_train)
    # Calculate the accuracy of the model
    trainscore.append(accuracy_score(y_train, predictions_train))
    testscore.append(accuracy_score(y_test, predictions))
    #coeffL2.append(model.coef_)
    
logc=np.linspace(-3, 3, num=13)
# Create the plot and set the label for each data series
plt.plot(logc, trainscore, label="l2train")
plt.plot(logc, testscore, label="l2test")
plt.xlabel("log_10(C)")
plt.ylabel("Accuracy")

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()  