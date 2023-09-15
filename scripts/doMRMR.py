"""
Script to select the top k features using the MRMR selection algorithm
"""

import numpy as np
import pandas as pd
import pickle
import pymrmr
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Take input and output files')
    parser.add_argument('--i', help='Input file containing data matrix (.csv) format', required=True)
    parser.add_argument('--o', help='Output file to store the top k features identified by MRMR', required=True)
    parser.add_argument('--k', help='Top k features to select using MRMR', required=True)
    parser.add_argument('--n', help='Number of bins to categorize data for MRMR', required=True)
    args = parser.parse_args()

    # load the normalized data
    input_file = args.i
    df = pd.read_csv(input_file)
    # drop the first column as it contains the sequence ID
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    shape = np.shape(df)
    # number of data points
    num_data_points = int(shape[0])
    # number of features
    num_features = int(shape[1])

    # Take out the category that has the class labels
    # for mrmr package, this needs to be the first column.
    labels = df.iloc[:, -1]
    df.drop(columns=df.columns[-1], axis=1, inplace=True)
    # list of column names
    var_name = list(df)

    # discretize the features and categorize them
    number_of_bins = int(args.n)
    for i in range(0, (num_features-1)):
        bins = np.linspace(np.min(df.iloc[:, i]), np.max(df.iloc[:, i]), number_of_bins)
        # go from continuous to categorical
        pd.cut(df.iloc[:, i], bins=bins)

    # add the labels as the first column of pd
    df.insert(0, 'labels', labels)

    # How many features to select
    num_features_to_select = int(args.k)

    # run mrmr
    score_indices = pymrmr.mRMR(df, 'MID', num_features_to_select)
    print(score_indices)

    output_file = args.o
    with open(output_file, 'wb') as f:
        pickle.dump(score_indices, f)
