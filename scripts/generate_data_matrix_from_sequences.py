"""
Script that reads through a collapsed data file with IDRs and pfam domains arranged by protein ID generated by
:mod:create_collapsed_data_file_of_pfam_domains_and_IDRs.py and generates a data matrix with labels
"""

import sys
sys.path.append("../utils/")
import os
import pandas as pd
import d2p2_sequence_analysis as d2p2
import data_generation_and_treatment as dgen


"""
Define input parameters to code
"""
# Input files
input_file_name = '../data/raw_sequence_data/collapsed_data_d2p2_nuclear_proteins.csv'

# Parameters to generate filtered sequences
linker_gap = 45
stitch_gap = 12
min_idr_length = 40
output_directory_sequence = '../data/filtered_sequence_data/'
output_sequence_file_name = 'filtered_sequences_linker_' + str(linker_gap) + '_stitch_' + str(stitch_gap) +'.fasta'

# Parameters to generate output data matrix file
output_directory_data_matrix = '../data/featurized_data_matrix/'
coarse_graining = 3
output_data_file_name = 'coarse_graining_' + str(coarse_graining) + '_new_aa_groups_data_matrix.csv'

# Parameters to generate features and labels
custom_aa_list = True
aa_group = [['E', 'D'], ['R', 'K'], ['Q', 'N', 'S', 'T', 'G', 'H', 'C'], ['A', 'L', 'M', 'I', 'V'], ['F', 'Y', 'W'],
            ['P']]
aa_group_names = ['NC', 'PC', 'Po', 'Apo', 'Aro', 'Pro']
motif_filename = '../data/motif_list/SLiMS-All.txt'
label_files = ['../data/protein_lists_in_compartments/TFs_Vaquerizas_Uniprot_Reviewed.txt',
               '../data/protein_lists_in_compartments/Transcriptional_coactivators_GO_list.txt']
label_legends = ['others', 'tfs', 'coactivators', 'overlap']

"""
Run script
"""

# Read data of IDRs and pfam domains from data file
df = pd.read_csv(input_file_name)
df_original = df.copy()

# Exclude IDRs present in pfam domains
d2p2.exclude_idrs_pfams(df)
# Stitch together IDRs that are spaced closer than stitch_gap
d2p2.stitch_idrs(df, stitch_gap)
# Write these filtered IDR sequences to a fasta file
try:
    # Create target Directory
    os.mkdir(output_directory_sequence)
    print("Directory ", output_directory_sequence, " Created ")
except OSError:
    print("Directory " + output_directory_sequence + " already exists")
d2p2.write_idr_file(df, output_directory_sequence + output_sequence_file_name)
filtered_filename = dgen.generate_filtered_file(output_directory_sequence + output_sequence_file_name, min_idr_length)

# Generate data matrix from sequences
flag_save_files = 1
indices_to_delete = []
use_all_features = 1
try:
    # Create target Directory
    os.mkdir(output_directory_data_matrix)
    print("Directory ", output_directory_data_matrix, " Created ")
except OSError:
    print("Directory " + output_directory_data_matrix + " already exists")
data_matrix, all_seqs_ids = dgen.generate_data_matrix([filtered_filename],
                                                      1,
                                                      motif_filename,
                                                      use_all_features,
                                                      coarse_graining,
                                                      output_directory_data_matrix,
                                                      indices_to_delete,
                                                      flag_save_files,
                                                      custom_aa_list=custom_aa_list,
                                                      aa_group=aa_group,
                                                      aa_group_names=aa_group_names)

# Assign labels to sequences in the data matrix
dgen.add_labels_to_data_matrix(data_matrix=data_matrix,
                               GO_list_files=label_files,
                               all_seqs_ids_in_DM=all_seqs_ids,
                               overlap_label=-1)
data_matrix.iloc[:, -1].values[data_matrix.iloc[:, -1].values == -1] = len(label_files) + 1
data_matrix.to_csv(output_directory_data_matrix + output_data_file_name)
