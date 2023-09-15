"""
Script that reads through raw data of IDRs and pfam domains generated by :mod:get_pfam_domains_and_IDRs_in_proteins.py
and stores it into a collapsed data file with the protein ID as the primary key
"""

import pandas as pd
import numpy as np
import argparse


def generate_collapsed_data_file(data_file, output_file, delimiter='\t'):

    largest_column_count = 0
    # Loop the data lines
    with open(data_file, 'r') as temp_f:
        # Read the lines
        lines = temp_f.readlines()
        for l in lines:
            # Count the column count for the current line
            column_count = len(l.split(delimiter)) + 1
            # Set the new most column count
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count

    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = [i for i in range(0, largest_column_count)]

    # Read csv
    df = pd.read_csv(data_file, header=None, delimiter=delimiter, names=column_names, low_memory=False)

    # Extract and collapse data pertaining to pfam domains
    pfam_indices = df.columns[7:]
    df['pfam_all'] = df[pfam_indices].astype(str).apply(lambda x: ','.join(x), axis=1)
    for count in range(len(df['pfam_all'].values)):
        df['pfam_all'].values[count] = df['pfam_all'].values[count].strip('nan,')
    less_data = df[list(df.columns[0:7]) + ['pfam_all']]
    less_data.columns = ['IDR_id', 'ID', 'sequence', 'description', 'idr_start', 'idr_end', 'gene', 'pfam_all']

    # Sort all data by sequence ID
    sorted_data = less_data.astype(str).groupby('ID').agg(lambda x: ','.join(x.unique()))
    length_proteins = np.array([float(len(x)) for x in sorted_data.iloc[:, 1]])
    sorted_data['lengths'] = length_proteins

    sorted_data.to_csv(output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Take input and output files')
    parser.add_argument('--i', help="Path to input file", required=True)
    parser.add_argument('--o', help="Path to output file", required=True)
    args = parser.parse_args()
    generate_collapsed_data_file(args.i, args.o)