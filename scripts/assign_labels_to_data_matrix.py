"""
Script that assigns labels to sequences in a data matrix depending on their category
"""

import argparse
import sys
import pandas as pd
sys.path.append("../utils/")
import data_generation_and_treatment as dgen


if __name__ == "__main__":

    label_files = ['../data/protein_lists_in_compartments/TFs_Vaquerizas_Uniprot_Reviewed.txt',
                   '../data/protein_lists_in_compartments/Transcriptional_coactivators_GO_list.txt']

    parser = argparse.ArgumentParser(description='Take input and output files')
    parser.add_argument('--i', help="Path to input file", required=True)
    parser.add_argument('--o', help="Path to output file", required=True)
    args = parser.parse_args()

    data_matrix_file = args.i
    data_matrix = pd.read_csv(data_matrix_file)

    # Assign labels to sequences in the data matrix
    dgen.add_labels_to_data_matrix(data_matrix=data_matrix,
                                   GO_list_files=label_files,
                                   all_seqs_ids_in_DM=data_matrix.iloc[:, 0],
                                   overlap_label=-1)
    data_matrix.iloc[:, -1].values[data_matrix.iloc[:, -1].values == -1] = len(label_files) + 1
    data_matrix.to_csv(args.o)
