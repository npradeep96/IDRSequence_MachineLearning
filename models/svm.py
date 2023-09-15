"""
Class that defines an SVM model
"""
import sklearn.model_selection
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


class SVM:

    def __init__(self):
        self.data_matrix = None
        self.train_data = None
        self.test_data = None
        self.model = None

    def load_data(self, data_matrix_file_name):
        """Loads data matrix present in data_matrix_file_name.csv """
        # Read the data matrix
        self.data_matrix = pd.read_csv(data_matrix_file_name)
        # Remove the sequence ID which is the first column of the data matrix
        self.data_matrix.drop(self.data_matrix.columns[0], axis=1, inplace=True)
        # Preprocess data
        self._preprocess_data()

    def split_train_test(self, test_size=0.25):
        """
        Function that splits data matrix into training and test data
        :return:
        """
        self.train_data, self.test_data = sklearn.model_selection.train_test_split(self.data_matrix,
                                                                                   test_size=test_size,
                                                                                   random_state=567)

    def _preprocess_data(self):
        """
        Function to preprocess data. For this, we are just mean centering and normalizing the data.
        :return:
        None
        """
        self.data_matrix.iloc[:, :-1] = StandardScaler().fit_transform(self.data_matrix.iloc[:, :-1])

    def build(self, kernel, gamma, c):
        """ Builds SVM model """
        self.model = SVC(kernel=kernel, gamma=gamma, C=c)

    def train(self):
        """
        Function to train the SVM model using training samples
        :return:
        train_accuracy(float): Balanced accuracy of the model on the training data
        """
        x_train = self.train_data.iloc[:, :-1]
        labels_train = self.train_data.iloc[:, -1]
        self.model.fit(x_train, labels_train)
        labels_predict_train = self.model.predict(x_train)
        train_accuracy = balanced_accuracy_score(labels_train, labels_predict_train)
        return train_accuracy

    def evaluate(self):
        """
        Function to evaluate model performance on the test data set
        :return:
        test_accuracy(float): Balanced accuracy of the model on the test data
        """
        x_test = self.test_data.iloc[:, :-1]
        labels_test = self.test_data.iloc[:, -1]
        y_pred = self.model.predict(x_test)
        test_accuracy = balanced_accuracy_score(labels_test, y_pred)
        return test_accuracy
