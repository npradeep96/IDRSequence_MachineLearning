"""
Script to perform PCA on a given data matrix and store the results
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle as pkl

# ref for code
# https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take input and output files')
    parser.add_argument('--i', help='Path to input data matrix file in (.csv) format', required=True)
    parser.add_argument('--o', help='Path to output file storing the PCA weights', required=True)
    args = parser.parse_args()

    # load the normalized data
    input_file = args.i
    df = pd.read_csv(input_file)

    shape = np.shape(df)
    num_data_points = int(shape[0])
    # number of data points
    num_features = int(shape[1])
    # number of features

    # separate out target from features
    X = df.iloc[:, 1:-1]
    # all but the first and last column are the features, because the first column is the sequence ID
    labels = df.iloc[:, -1]
    # last column is the labels

    # standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA().fit(X_scaled)

    pca_data = {'components': pca.components_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'mean': pca.mean_
                }

    # Output data to file
    output_file = args.o
    with open(output_file, 'wb') as output:
        pkl.dump(pca_data, output)
