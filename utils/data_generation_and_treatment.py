#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:37:04 2019
cd
@author: pradeep / modifications by cecilia

"""

"""
File containing functions required for the "__Clean_code__" Jupyter Notebook that:

    1- Generates the files to be used as data (TF, CO and Others) from the data from Uniprot and GO
    2- Constructs the data matrix to be used in the analysis
    3- Performs the analysis (PCA, tSNE, cumulative var, hierarchical clustering, fold enrichment...)

"""

### ALL IMPORTS NEEDED FOR THE FOLLOWING FUNCTIONS:

from Bio import SeqIO
import os
import class_sequence as cs
import numpy as np
import csv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cluster_tools as ct
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from operator import itemgetter
import xlsxwriter


###############################################################################
#                                       1                                     #
###############################################################################

""" GENERATING FASTA FILES FOR CO, TF AND OTHERS FROM UNIPROT AND GO DATA """

def generate_filtered_file(filter_filename, min_aa_length):
    save_filename = filter_filename.split('.fasta')[0] + '_filtered.fasta'
    seq_records = SeqIO.parse(filter_filename,'fasta')

    def filtered_sequences(): #filters seqs lesser than 40 AAs long from IDR data set
        for record in seq_records:
            if len(record) > min_aa_length:
                yield record

    iterable = filtered_sequences()
    SeqIO.write(iterable,save_filename,'fasta')
    return save_filename


def generate_CO_TF_ONP_fasta_files():

    # -------------- Need to change this section accordingly -----------------
    COGO_name = "raw data - Uniprot and GO/Coactivator_GO_list.txt"
    TFGO_name = "raw data - Uniprot and GO/TFs_Vaquerizas_Uniprot_Reviewed.txt"
    filtered_filename = "raw data - Uniprot and GO/human_IDR_proteome_filtered.fasta"
    write_CO = "pre treated data for analysis/Coactivator-IDRs.fasta"
    write_TF = "pre treated data for analysis/TF-IDRs.fasta"
    write_nonCOTF = "pre treated data for analysis/nonCOTF-IDRs.fasta"
    write_COTF = "pre treated data for analysis/COTF-IDRs.fasta"

    # ------------------------------------------------------------------------

    COGO = open(COGO_name,"r")

    TFGO = open(TFGO_name,"r")

    TF_IDs = list()
    # List of transcription factor accession numbers

    for TF in TFGO:
        TF_IDs.append(str(TF.rstrip()))

    coactivator_IDs = list()
    # List of coactivator accession numbers

    for coactivator in COGO:
        if coactivator.rstrip() not in TF_IDs:
            coactivator_IDs.append(str(coactivator.rstrip()))

    # Extract out lists of TFs, coactivators and other IDRs in separate lists

    TF_sequences = list()
    CO_sequences = list()
    nonCOTF_sequences = list()

    for sequence in SeqIO.parse(filtered_filename,'fasta'):
        if str(sequence.id[0:6]) in TF_IDs:
            TF_sequences.append(sequence)
        elif str(sequence.id[0:6]) in coactivator_IDs:
            CO_sequences.append(sequence)
        else:
            nonCOTF_sequences.append(sequence)

    # Write the above extracted lists to appropriate files
    SeqIO.write(TF_sequences,write_TF,'fasta')
    SeqIO.write(CO_sequences,write_CO,'fasta')
    SeqIO.write(nonCOTF_sequences,write_nonCOTF,'fasta')
    SeqIO.write(TF_sequences + CO_sequences,write_COTF,'fasta')

    print('The number of TF IDR sequences is:',len(TF_sequences))
    print('The number of coactivator IDR sequences is:',len(CO_sequences))
    #print('The number of coactivator/TF IDR sequences is:',len(TF_sequences)+len(CO_sequences))
    print('The number of non- coactivator/TF IDR sequences is:',len(nonCOTF_sequences))

    COGO.close()
    TFGO.close()

def keep_only_nuclear_prots(data_directory, nuclear_prots_file, comparison_file):

    nb_nuclear_only = 0
    nb_non_nuclear_only = 0
    intersection_length = 0

    compare_list = list()

    for sequence in SeqIO.parse(comparison_file,'fasta'):
        compare_list.append(sequence.id[0:6])

    nuclear_list = list()

    for sequence in SeqIO.parse(nuclear_prots_file,'fasta'):
        nuclear_list.append(sequence.id[3:9])

    intersection_length = len(list(set(compare_list) & set(nuclear_list)))

    nb_nuclear_only = len(nuclear_list) - intersection_length
    nb_non_nuclear_only = len(compare_list) - intersection_length

    #print("The number of %s proteins that localize in the nucleus is: %s" %(comparison_file.split('/')[1].split('.fasta')[0], intersection_length))

    IDRs_to_consider = list(set(compare_list) & set(nuclear_list))

    selected_nuclear_IDRs = list()

    for sequence in SeqIO.parse(comparison_file,'fasta'):
        if sequence.id.split('_')[0] in IDRs_to_consider:
            selected_nuclear_IDRs.append(sequence)

    SeqIO.write(selected_nuclear_IDRs, comparison_file.split('.fasta')[0] + '-culled.fasta',"fasta")

    with open(comparison_file.split('.fasta')[0] + "-culled-IDs.txt", 'w') as f:
        for item in IDRs_to_consider:
            f.write("%s\n" % item)
    f.close()

    print("The number of \'%s\' IDRs that localize in the nucleus is: %s" %(comparison_file.split('/')[1].split('.fasta')[0],len(selected_nuclear_IDRs)))



###############################################################################
#                                       2                                     #
###############################################################################
""" GENERATING DATA MATRIX """


### Function to create a new directory if it doesn't exist

def create_dir(file_path):

    try:
        os.makedirs(os.path.dirname(file_path))
        print("Directory " + file_path +  " Created ")
    except OSError:
        print("Directory " + file_path +  " already exists")


########### function to count motifs from sequence files #######################

def count_motifs(Seq_filename, Motif_filename):

    """
    In this function, I have assumed that the motifs are stored in a file where each line is of the form:

    motif-name, motif-regex
    """

    list_of_motif_counts = list()

    for IDR_seq in SeqIO.parse(Seq_filename, 'fasta'):

        my_seq = cs.sequence(str(IDR_seq.seq))
        my_seq.read_motifs(Motif_filename)
        my_seq.count_motifs()

        # print my_seq.motif_count
        list_of_motif_counts.append(my_seq.motif_count)

    return list_of_motif_counts

def delete_non_occuring_motifs(Motif_filename, fasta_file_list):
    # Deleting the motifs that don't even occur once
    
    list_of_motifs = list()
    colsum = list()
    i=0
    
    for f in fasta_file_list:
        list_of_motifs.append(count_motifs(f, Motif_filename))
        colsum.append(np.sum(np.array(list_of_motifs[i]),axis=0))
        i += 1

    motif_file = open(Motif_filename,"r+")

    lines_from_motif_file = motif_file.readlines()
    motif_names = list()
    for motif in lines_from_motif_file:
        motif_names.append(motif.split(',')[0])
    
    for i in range(len(fasta_file_list)):
        idx = np.where(colsum[i] == 0)
        if i == 1:
            print('Motifs having counts = 0 across the %s st file : %s' %(i, np.array(motif_names)[idx]))
        elif i == 2:
            print('Motifs having counts = 0 across the %s nd file : %s' %(i, np.array(motif_names)[idx]))
        elif i == 3:
            print('Motifs having counts = 0 across the %s rd file : %s' %(i, np.array(motif_names)[idx]))
        else:
            print('Motifs having counts = 0 across the %s th file : %s' %(i, np.array(motif_names)[idx]))
        
    
    indices_to_delete = np.union1d(np.where(no_appearance == 0) for no_appearance in colsum)

    return indices_to_delete


### Function to create a consolidated data matrix

def create_consolidated_data_matrix(files, write_file_path):

    outpath = write_file_path
    fout = open(outpath + "consolidated_data_matrix.csv","w")

    # first file:
    count = 0;
    for line in open(files[0]):
        fout.write(line)
        count = count+1;
    # now the rest:
    count = 0;
    for filename in files[1:]:
        with open(filename,'r') as f:
            next(f)
            for line in f:
                if count >0:
                    fout.write(line)
                count = count+1
            f.close()

    fout.close()


### Function to generate the data matrix ######################################

def generate_data_matrix(list_fasta_files, label, Motif_filename, use_all_features, coarse_graining, write_file_path, indices_to_delete, flag_save_files, custom_aa_list=False, aa_group=None, aa_group_names=None):
        
    if flag_save_files:
        dirName = write_file_path
    
        try:
        # Create target Directory
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        except:
            print("Directory " + dirName +  " already exists")
    
        outfile_name = []
        
        i=0
        for filename in list_fasta_files:
            outfile_name.append(dirName + '/' + filename.split('/')[-1].rstrip('.fasta') + '-features.csv')
            writeFile = open(outfile_name[i], 'w')
            writer = csv.writer(writeFile)
    
            dummy_seq = cs.sequence(custom_aa_list=custom_aa_list,aa_group=aa_group,aa_group_names=aa_group_names)
            dummy_seq.read_motifs(Motif_filename)
    
            feature_name_list = list()
            feature_name_list.append('sequence id')

            all_seqs_list = list()

            if use_all_features:
    
                # Frequencies of the different groups
                for grp_name in dummy_seq.aa_group_names:
                    feature_name_list.append('f-' + grp_name )
    
                # Features based on block size distribution
                for grp_name in dummy_seq.aa_group_names:
                    for feature_name in dummy_seq.names_features_from_bsd:
                        feature_name_list.append(grp_name + '-' + feature_name)
    
                for grp_name1 in dummy_seq.aa_group_names:
                    for grp_name2 in dummy_seq.aa_group_names:
                        for feature_name in dummy_seq.names_features_from_bsd:
                            feature_name_list.append(grp_name1 + '-' + grp_name2 + '-' + feature_name)
    
            # Features based on motifs
    
            for motif_name in dummy_seq.motif_name_list:
                feature_name_list.append(motif_name)

            feature_name_list.append('length')
            # Label telling whether the sequence is a TF, CO or other
            feature_name_list.append('label')
    
            # print feature_name_list
    
            writer.writerows([feature_name_list])
    
            # Now, the list feature_name_list has the names of all the features
    
            for IDR_sequence in SeqIO.parse(filename, 'fasta'):
    
                all_seqs_list.append(IDR_sequence.id)
    
                list_of_features = list()
                list_of_features.append(IDR_sequence.id)
    
                my_seq = cs.sequence(str(IDR_sequence.seq))
    
                if use_all_features:
    
                    my_seq.get_composition()
                    list_of_features += my_seq.composition
    
                    my_seq.smear_sequence(coarse_graining)
                    my_seq.get_single_bsd()
                    my_seq.get_pairwise_bsd()
                    my_seq.get_features_from_bsd()
                    list_of_features += [item for sublist in my_seq.features_from_bsd for item in sublist]
                # This is necessary because I need to flatten out my_seq.features_from_bsd, which is a 2-D list
    
                my_seq.read_motifs(Motif_filename)
                my_seq.count_motifs()
                motif_list = list(np.delete(my_seq.motif_count, indices_to_delete))
                list_of_features += motif_list
                list_of_features += [len(IDR_sequence)]
    
                list_of_features += [label]
    
                writer.writerows([list_of_features])
            i+=1
    
        writeFile.close()
    
        create_consolidated_data_matrix(outfile_name, write_file_path)
        data_matrix = pd.read_csv(write_file_path + "/consolidated_data_matrix.csv")
        data_matrix = data_matrix.set_index('sequence id')
        
    else:
        
        feature_matrix = list()
        all_seqs_list = list()
        
        dummy_seq = cs.sequence()
        dummy_seq.read_motifs(Motif_filename)

        feature_name_list = list()
        feature_name_list.append('sequence id')

        if use_all_features:

            # Frequencies of the different groups
            for grp_name in dummy_seq.aa_group_names:
                feature_name_list.append('f-' + grp_name )

            # Features based on block size distribution
            for grp_name in dummy_seq.aa_group_names:
                for feature_name in dummy_seq.names_features_from_bsd:
                    feature_name_list.append(grp_name + '-' + feature_name)

            for grp_name1 in dummy_seq.aa_group_names:
                for grp_name2 in dummy_seq.aa_group_names:
                    for feature_name in dummy_seq.names_features_from_bsd:
                        feature_name_list.append(grp_name1 + '-' + grp_name2 + '-' + feature_name)

        # Features based on motifs

        for motif_name in dummy_seq.motif_name_list:
            feature_name_list.append(motif_name)

        feature_name_list.append('length')
        # Label telling whether the sequence is a TF, CO or other
        feature_name_list.append('label')
        
        
        
        for filename in list_fasta_files:
    
            # Now, the list feature_name_list has the names of all the features
    
            for IDR_sequence in SeqIO.parse(filename, 'fasta'):
    
                all_seqs_list.append(IDR_sequence.id)
    
                list_of_features = list()
                list_of_features.append(IDR_sequence.id)
    
                my_seq = cs.sequence(str(IDR_sequence.seq))
    
                if use_all_features:
    
                    my_seq.get_composition()
                    list_of_features += my_seq.composition
    
                    my_seq.smear_sequence(coarse_graining)
                    my_seq.get_single_bsd()
                    my_seq.get_pairwise_bsd()
                    my_seq.get_features_from_bsd()
                    list_of_features += [item for sublist in my_seq.features_from_bsd for item in sublist]
                # This is necessary because I need to flatten out my_seq.features_from_bsd, which is a 2-D list
    
                my_seq.read_motifs(Motif_filename)
                my_seq.count_motifs()
                motif_list = list(np.delete(my_seq.motif_count, indices_to_delete))
                list_of_features += motif_list

                list_of_features += [len(IDR_sequence)]
    
                list_of_features += [label]
    
                feature_matrix.append([IDR_sequence.id]+list_of_features)
    
    
        data_matrix = pd.DataFrame(data=feature_matrix,columns=feature_name_list)
        data_matrix = data_matrix.set_index('sequence id')
    
    return data_matrix, all_seqs_list



### Function to add labels to the data_matrix by reading a list of GO_id files (.txt)

def add_labels_to_data_matrix(data_matrix, GO_list_files, all_seqs_ids_in_DM, overlap_label=-1):

    """
    This function takes in data matrix (sequences * features) , list of GO_id
    labels, list of sequence IDS, and value for overlap label.

    In the way this function is written, if there is overlap in the sequence
    ids of 2 or more files, the row is labelled as -1 as default or as the label
    entered by the user
    """

    for x in set(data_matrix.iloc[:,-1]):
        data_matrix.iloc[:,-1] = data_matrix.iloc[:,-1].replace(x,0)

    j=0
    for GO_ids_file in GO_list_files:
        j += 1
        if GO_ids_file.find('xlsx') > -1:
            pos = pd.read_excel(GO_ids_file);
            GO_ids = list(pos['Entry'].values);

        else:
            filename = open(GO_ids_file, 'r')
            GO_ids = [x.split('\\')[0].strip() for x in filename]
            filename.close()

        #we filtered our protein list! ==> not all the proteins from the condensate file will appear in our all_seqs_ids
        for i in range(data_matrix.shape[0]):
            if all_seqs_ids_in_DM[i].split('_')[0] in GO_ids:
                if data_matrix.iloc[i,-1] != 0:
                    data_matrix.iloc[i,-1] = overlap_label
                else:
                    data_matrix.iloc[i,-1] = j

    return data_matrix



###############################################################################
#                                       3                                     #
###############################################################################
""" ANALYSIS OF THE DATA (PLOTS) """


### Function to make plots pretty

def make_nice_axis(ax):
    """ Function to beautify axis, based on version written by D. Murakowski"""

    ax.spines['top'].set_visible(False) # hide top axs
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 30))
    ax.spines['left'].set_position(('outward', 30))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(pad=10)
    ax.yaxis.set_tick_params(pad=10)
    ax.xaxis.set_tick_params(labelsize=32)
    ax.yaxis.set_tick_params(labelsize=32)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 20


### Get feature pairwise correlation matrix

def get_pairwise_corr(write_file_path, data_matrix, flag_save_files):

    data_matrix_without_labels = data_matrix.iloc[:,0:-1]
    # Remove the last column that contains labels

    corr_mat = data_matrix_without_labels.corr(method='pearson')
    plt.figure(figsize=(20,10))
    ax = sns.heatmap(corr_mat,vmax=1,square=True,cmap='cubehelix')

    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    
    if flag_save_files:
        figure_path = write_file_path + '/figures'
        file_path = figure_path + '/Feature_corr_matrix'
    
        create_dir(file_path)
    
        plt.savefig(file_path+'.svg',format='svg',dpi=600,bbox_inches='tight')
        plt.savefig(file_path+'.png',format='png',dpi=600,bbox_inches='tight')
    
    plt.show()
    plt.close()

    return corr_mat

### Function to perform dimensionality reduction

def dimensionality_reduction(write_file_path, data_matrix, flag_save_files, switch_PCA_tSNE):

    data_matrix_without_labels = data_matrix.iloc[:, :-1]
    true_labels = np.array( data_matrix.iloc[:, -1] )
    figure_path = write_file_path + '/figures'

    data_matrix_norm = StandardScaler().fit_transform( np.asarray(data_matrix_without_labels.values) )

    if switch_PCA_tSNE == 1:

        pc_obj = PCA().fit(data_matrix_norm)
        pc_scores = PCA().fit_transform(data_matrix_norm)

        reduced_data_matrix = pd.DataFrame(pc_scores)
        
        if flag_save_files:
            reduced_data_matrix.to_csv(write_file_path + "/data_matrix/reduced_data_matrix_PCA.csv")

        ### Plotting cumulative explained variance

        fig, ax = plt.subplots(figsize=(20,16))
        make_nice_axis(ax)

        explained_variance = np.cumsum(pc_obj.explained_variance_ratio_)*100
        x_ticks = [i for i in range(1,31)]

        ax.bar(np.arange(len(x_ticks)), explained_variance[0:len(x_ticks)], align='center', alpha=0.3)

        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_xticklabels(x_ticks, fontsize=15)
        ax.set_ylabel('Explained variance', fontsize=40)
        ax.set_title('Cumulative % variance by PCs', fontsize=40)
        
        if flag_save_files:
            file_path = figure_path + "/PCA/cumulative_explained_variance"
            create_dir(file_path)
            plt.savefig(file_path+'.svg',format='svg',dpi=600,bbox_inches='tight')
            plt.savefig(file_path+'.png',format='png',dpi=600,bbox_inches='tight')
        
        plt.show()
        plt.close()

        ### Plotting percent variance explained by each feature

        fig, ax = plt.subplots(figsize=(20,16))
        make_nice_axis(ax)

        explained_variance = np.array(pc_obj.explained_variance_ratio_)*100
        cutoff = 100/len(pc_obj.explained_variance_ratio_)
        x_ticks = [i for i in range(1,16)]

        ax.bar(np.arange(len(x_ticks)), explained_variance[0:len(x_ticks)], align='center', alpha=0.3)
        plt.plot(x_ticks, 15*[cutoff], 'r--')

        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_xticklabels(x_ticks, fontsize=20)
        ax.set_ylabel('Explained variance', fontsize=40)
        ax.set_title('% variance by PCs', fontsize=40)
        
        if flag_save_files:
            file_path = figure_path + "/PCA/explained_variance"
            create_dir(file_path)
            plt.savefig(file_path+'.svg',format='svg',dpi=600,bbox_inches='tight')
            plt.savefig(file_path+'.png',format='png',dpi=600,bbox_inches='tight')
        
        plt.show()
        plt.close()

        ### Plotting the top features by weight along the first principal component

        pc_comp = 0
        first_N = 15

        key = np.argsort(abs(pc_obj.components_[pc_comp,:]))[::-1]
        sorted_vec, sorted_labels = np.abs(pc_obj.components_[pc_comp,key]), data_matrix.columns[key]
        random_vec = np.ones(len(sorted_vec))/pow(len(sorted_vec),0.5)

        fig,ax = plt.subplots(figsize=(20,16))
        make_nice_axis(ax)

        x = np.arange(len(data_matrix.columns[:first_N]));

        ax.bar(x, sorted_vec[x], align='center', alpha=0.5)
        ax.plot(x, random_vec[x],'r--')

        plt.xticks(x, sorted_labels, fontsize = 20, rotation='vertical')
        ax.set_ylabel('Relative weight of eigen vector', fontsize=40)
        ax.set_title('Top weights on pc_{}'.format(pc_comp+1), fontsize=40)
        
        if flag_save_files:
            file_name =  figure_path +'/PCA/Weights_pc_' + str(pc_comp+1) + '_Nfeat_' + str(first_N)
            create_dir(file_name)
            plt.savefig(file_name+'.svg',format='svg',dpi=600,bbox_inches='tight')
            plt.savefig(file_name+'.png',format='png',dpi=600,bbox_inches='tight')

        plt.show()

        return pc_scores, true_labels

    else:

        data_matrix_for_clustering = ct.tSNE_projection(data_matrix_norm)
        reduced_data_matrix = pd.DataFrame(data_matrix_for_clustering)
        
        if flag_save_files:
            reduced_data_matrix.to_csv(write_file_path + "/data_matrix/reduced_data_matrix_tSNE.csv")

        return data_matrix_for_clustering, true_labels


### Function to plot the 2-D projections of the dataset in PCA and tSNE space

def plot_2D(write_file_path, pc_scores, true_labels, label_legends, colors, flag_save_files, switch_PCA_tSNE, pc_int1=1, pc_int2=2, x_lims=0, y_lims=0):

    figure_path = write_file_path + '/figures'

    figure_pca,ax = plt.subplots(figsize=(20,16))
    make_nice_axis(ax)

    for p in range(len(label_legends)):
        count = len(label_legends)-1-p;
        rel_indices = np.where(true_labels==count+1)[0];
        if (count<len(label_legends)-1):
            ax.scatter(pc_scores[rel_indices,pc_int1],pc_scores[rel_indices,pc_int2],color=colors[count],s=50,label=label_legends[count])
        else:
            ax.scatter(pc_scores[rel_indices,pc_int1],pc_scores[rel_indices,pc_int2],color=colors[count],alpha=0.5,s=50,label=label_legends[count])

    ax.legend(fontsize=25)
    ax.set_xlabel('Component '+ str(pc_int1+1),fontsize=40)
    ax.set_ylabel('Component '+ str(pc_int2+1),fontsize=40)

    if x_lims:
        ax.set_xlim(x_lims)
    if y_lims:
        ax.set_ylim(y_lims)

    if switch_PCA_tSNE == 1:
        ax.set_title('Principal Component Analysis',fontsize=40)
        
        if flag_save_files:
            file_path =  figure_path + '/PCA/pcs_' + str(pc_int1+1) + '_' + str(pc_int2+1)
            create_dir(file_path)
            figure_pca.savefig(file_path+'.svg',format='svg',dpi=600,bbox_inches='tight')
            figure_pca.savefig(file_path+'.png',format='png',dpi=600,bbox_inches='tight')
    else:
        ax.set_title('tSNE Analysis',fontsize=40)
        
        if flag_save_files:
            file_path =  figure_path + '/tSNE/pcs_' + str(pc_int1+1) + '_' + str(pc_int2+1)
            create_dir(file_path)
            figure_pca.savefig(file_path+'.svg',format='svg',dpi=600,bbox_inches='tight')
            figure_pca.savefig(file_path+'.png',format='png',dpi=600,bbox_inches='tight')

    plt.show()
    

### Function to cluster reduced data matrix

def cluster_reduced_matrix(reduced_data_matrix, true_labels, num_components, num_clusters, write_file_path, flag_save_files, switch_PCA_tSNE):

    reduced_data_matrix = np.delete(reduced_data_matrix, range(num_components, np.size(reduced_data_matrix,1)), 1)
    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(reduced_data_matrix)

    clust_labels = ward.labels_ + 1

    num_labels = np.max(true_labels) + 1

    cluster_purity_counter = np.zeros([num_clusters, num_labels])
    rep_label = np.zeros(num_clusters)

    for i in range(0,num_clusters):

        max_val = 0

        for j in range(0,num_labels):

            cluster_purity_counter[i][j] += len(np.where(true_labels[list(np.where(clust_labels == i + 1)[0])] == j)[0])

            if cluster_purity_counter[i][j] >= max_val:
                max_val = cluster_purity_counter[i][j]
                rep_label[i] = j + 1

    if flag_save_files:
        clustering_path = write_file_path + "/clustering_results/Cluster_Purity_"

        create_dir(clustering_path)

        np.savetxt(clustering_path + str(switch_PCA_tSNE) + '_' + str(num_components) + "_" + str(num_clusters) + ".csv", cluster_purity_counter.transpose(), delimiter=',')

    return clust_labels


### Plotting cluster label coloured plots

def plot_2D_cluster(write_file_path, pc_scores, labels, num_components, flag_save_files, switch_PCA_tSNE, pc_int1=0, pc_int2=1, x_lims=0, y_lims=0):

    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    keys = np.argsort(counts_labels)[::-1]

    if flag_save_files:
        figure_path = write_file_path + '/figures'

    new_labels = np.array(labels)

    for i in range(0,len(keys)):
        new_labels[np.where(labels == unique_labels[keys[i]])] = i+1

    labels = np.array(new_labels)

    figure_pca,ax = plt.subplots(figsize=(20,16))
    make_nice_axis(ax)

    label_legends = ['C %s' %i for i in range(1,max(labels)+1)]
    colors = (cm.rainbow(np.linspace(0, 1, max(labels))))

    for p in range(len(label_legends)):
        count = len(label_legends)-p-1;
        rel_indices = np.where(labels==count+1)[0];
        ax.scatter(pc_scores[rel_indices,pc_int1],pc_scores[rel_indices,pc_int2],color=colors[count],s=50,label=label_legends[count])

    ax.legend(fontsize=25)
    ax.set_xlabel('Component '+ str(pc_int1+1),fontsize=40)
    ax.set_ylabel('Component '+ str(pc_int2+1),fontsize=40)

    if x_lims:
        ax.set_xlim(x_lims)
    if y_lims:
        ax.set_ylim(y_lims)

    if switch_PCA_tSNE == 1:
        ax.set_title('Principal Component analysis',fontsize=40)
        
        if flag_save_files:
            file_path =  figure_path + '/PCA/pcs_' + str(pc_int1+1) + '_' + str(pc_int2+1)
            create_dir(file_path)
            figure_pca.savefig(file_path+'_'+str(num_components)+"_"+str(max(labels))+'clusters.svg',format='svg',dpi=600,bbox_inches='tight')
            figure_pca.savefig(file_path+'_'+str(num_components)+"_"+str(max(labels))+'clusters.png',format='png',dpi=600,bbox_inches='tight')
    else:
        ax.set_title('tSNE analysis',fontsize=40)
        
        if flag_save_files:
            file_path =  figure_path + '/tSNE/pcs_' + str(pc_int1+1) + '_' + str(pc_int2+1)
            create_dir(file_path)
            figure_pca.savefig(file_path+'_'+str(num_components)+"_"+str(max(labels))+'clusters.svg',format='svg',dpi=600,bbox_inches='tight')
            figure_pca.savefig(file_path+'_'+str(num_components)+"_"+str(max(labels))+'clusters.png',format='png',dpi=600,bbox_inches='tight')

    plt.show()


def generate_purity_plots(write_file_path, label_legends, flag_save_files, switch, num_components, num_clusters):

    filename = write_file_path + "/clustering_results/Cluster_Purity_" + str(switch) + "_" + str(num_components) + "_" + str(num_clusters) + ".csv"
    figure_path = write_file_path + '/figures'

    my_data = np.genfromtxt(filename, delimiter=',')
    colsum = np.sum(my_data, 0)
    normalized_data = np.divide(my_data, colsum);
    # print my_data
    # print my_data/np.sum(my_data,0)
    # print np.sum(my_data, 1)/np.sum(my_data)
    enrichment = (my_data/np.sum(my_data,0)).transpose()/(np.sum(my_data, 1)/np.sum(my_data))

    ### Plotting cluster composition

    figure_pca,ax = plt.subplots(figsize=(20,16))
    make_nice_axis(ax)

    ind = np.linspace(1, np.size(normalized_data,1), np.size(normalized_data,1))
    key = np.argsort(colsum)[::-1]
    sorted_vec, sorted_labels, sorted_enrichment = normalized_data[:,key], colsum[key], enrichment[key, :]
    
#    bottom_vec = []
#    bottom_vec.append([])
#    for label in label_legends:
#        i = label_legends.index(label)
#        bottom_vec.append(np.concatenate((bottom_vec[i],sorted_vec[i,:]))
    
    i = len(label_legends)-1
    while i>0:
        plt.bar(ind, sorted_vec[i,:], label=label_legends[i], bottom = sum(sorted_vec[a,:] for a in range(0,i)), align='center', alpha=0.3)
        i -= 1
    plt.bar(ind, sorted_vec[0,:], label=label_legends[0], align='center', alpha=0.3)

    plt.xticks(ind, sorted_labels, fontsize = 100/num_clusters + 10)
    plt.ylabel("Cluster Composition", fontsize = 40)
    plt.xlabel("Cluster Size", fontsize = 40)
    plt.legend(loc="upper right", fontsize = 25)

    if switch == 1:
        method = 'PCA'
    else:
        method = 't-SNE'

    plt.title("Cluster Composition for " + method + ", components = " + str(num_components), fontsize = 40)
    
    if flag_save_files:
        file_path = figure_path + "/clustering_results/Cluster_purity_"
        create_dir(file_path)
        plt.savefig(file_path + method + '_' + str(num_components) + '_' + str(num_clusters) + '.svg',format='svg',dpi=600,bbox_inches='tight')
        plt.savefig(file_path + method + '_' + str(num_components) + '_' + str(num_clusters) + '.png',format='png',dpi=600,bbox_inches='tight')
    
    plt.show()
    plt.close()

    ### Plotting fold enrichment


    for enrichment_of_protein in range(1,len(label_legends)+1):

        fig, axes = plt.subplots(2,1,figsize=(25,20),sharex=True)
        make_nice_axis(axes[0])
        make_nice_axis(axes[1])

        labels = ['C %s' %i for i in range(1,num_clusters+1)]

        x = np.arange(len(labels))

        axes[0].bar(x, sorted_labels, align='center', alpha=0.3)
        axes[1].bar(x, sorted_enrichment[:,enrichment_of_protein-1], align='center', alpha=0.3)
        axes[1].plot(x,np.ones((len(labels),1)),'r--')
        axes[1].set_xticks(x)
        axes[0].set_xticklabels(labels,fontsize = 100/num_clusters + 10)
        axes[1].set_xticklabels(labels,fontsize = 100/num_clusters + 10)
        axes[0].set_ylabel('Cluster size', fontsize=30)
        axes[1].set_ylabel('Fold enrichment',fontsize=30)
        axes[0].set_yscale('log')
        axes[1].set_title('Cluster enrichment of ' + label_legends[enrichment_of_protein-1],fontsize=30)
        axes[0].set_title("Cluster Enrichment for " + method + ", components = " + str(num_components), fontsize = 30)

        if flag_save_files:
            file_path = figure_path + "/clustering_results/Cluster_enrichment_"
            create_dir(file_path)
            plt.savefig(file_path + label_legends[enrichment_of_protein-1] + '_' + method + '_' + str(num_components) + '_' + str(num_clusters) + '.svg',format='svg',dpi=600,bbox_inches='tight')
            plt.savefig(file_path + label_legends[enrichment_of_protein-1] + '_' + method + '_' + str(num_components) + '_' + str(num_clusters) + '.png',format='png',dpi=600,bbox_inches='tight')
        
        plt.show()
        plt.close()

    return enrichment


### Function to find ids of proteins in a particular cluster

def get_prot_id_from_clust(write_file_path, data_matrix, label_legends, clust_labels, all_seqs_ids, num_components, num_clusters, switch_PCA_tSNE):

    create_dir(write_file_path+"/proteins in each cluster/")
    
    if switch_PCA_tSNE == 1:
        ids_filename = str(write_file_path) + '/proteins in each cluster/PCA_' + str(num_components) + '_' + str(num_clusters) + 'clusters'
    elif switch_PCA_tSNE == 2:
        ids_filename = str(write_file_path) + '/proteins in each cluster/tSNE_2_' + str(num_clusters) + 'clusters'
    
    newfile = xlsxwriter.Workbook(ids_filename + '.xlsx')
    
#     std_row = ['Uniprot ID','Gene name','Gene description','IDR number','Start position','End position']
    prot_labels = data_matrix.iloc[:,-1]
    min_DM_label = min(set(prot_labels))
    max_DM_label = max(set(prot_labels))                
                
                

    for clust_of_interest in range(1, num_clusters+1): 
        
        clust_of_int_ind = np.array(list(np.where(clust_labels==clust_of_interest)[0]))
        worksheet = newfile.add_worksheet('Cluster ' + str(clust_of_interest))
        row = 0
        worksheet.write(row, 0, 'cluster size: ' + str(len(clust_of_int_ind)))

        label = max_DM_label
        while label >= min_DM_label:
                        
            prot_lab_ind = np.array(list(np.where(prot_labels==label)[0]))
            prot_lab_in_clust = list(set(clust_of_int_ind)&set(prot_lab_ind))
            prot_in_clust_ids = list(data_matrix.index[prot_lab_in_clust])
        
            row +=2
            
            if label>=0:
                worksheet.write(row, 0, label_legends[label])
                worksheet.write(row, 1, len(prot_in_clust_ids))
                worksheet.write(row, 2, '(nb of seqs belonging to this label)')
                
            elif i<0:
                worksheet.write(row, 0, 'overlap')
                worksheet.write(row, 1, len(prot_in_clust_ids))
                worksheet.write(row, 2, '(nb of seqs belonging to this label)')
                
                
            row +=1
            worksheet.write(row, 0, 'Uniprot ID')
            worksheet.write(row, 1, 'Gene name')
            worksheet.write(row, 2, 'Gene description')
            worksheet.write(row, 3, 'IDR number')
            worksheet.write(row, 4, 'Start position')
            worksheet.write(row, 5, 'End position')
                
            for j in prot_in_clust_ids:
                row +=1
                worksheet.write(row, 0, j.split('_')[0])
                worksheet.write(row, 1, j.split('|')[1])
                worksheet.write(row, 2, j.split('|')[2])
                worksheet.write(row, 3, j.split('_')[1])
                worksheet.write(row, 4, j.split('_')[2])
                worksheet.write(row, 5, j.split('_')[3].split('|')[0])

            row +=1
            
            label -= 1
        
    newfile.close()
### Get fold enrichment of features

def get_feat_fold_enrichment(data_matrix, clust_labels, num_clusters, switch_PCA_tSNE):

    feature_names =  list(data_matrix.columns)
    feature_names.remove('label')
    mean_dataset = [data_matrix[feat_name].mean() for feat_name in feature_names]
    feat_fold_enrichment = dict()
    
    for clust in range(1,num_clusters+1):
        
        clust_of_interest = list(np.where(clust_labels == clust)[0])
        mean_cluster = data_matrix.iloc[clust_of_interest,:].mean(axis = 0)
    
        fold_enrichment = []
        for i in range(len(mean_dataset)):
            fold_enrichment.append([mean_cluster[i]/mean_dataset[i], feature_names[i]])
    
        feat_fold_enrichment['C'+ str(clust)]=sorted(fold_enrichment, key=itemgetter(0))[::-1]

    return feat_fold_enrichment



# Function to plot fold enrichment of labels vs cluster size (in % of overall dataset)

def plot_enrichment_vs_cluster_size(write_file_path, output_filename, data_matrix, labels, label_legends, colors,  xlim, ylim, switch, flag_save_files, num_components, num_clusters):

    '''
    This function assumes that the data matrix was created using the previous functions
    and therefore is constructed with the last column corresponding to the labels of the data set

    The label_list list has to be entered in the same order as the labelling in the data_matrix:
        example:
            ['Coactivator', 'TF', 'Others']

        if we have that the label 1 in the data_matrix is attributed to 'Coactivator', 2 to 'TF' and so on
        ---> same reasoning for colors
    '''

    label_counts = data_matrix.groupby('label').size()
    frac = {}

    filename = write_file_path + "/clustering_results/Cluster_Purity_" + str(switch) + "_" + str(num_components) + "_" + str(num_clusters) + ".csv"
    my_data = np.genfromtxt(filename, delimiter=',')
    colsum = np.sum(my_data, 0)
    normalized_data = np.divide(my_data, colsum);
    enrichment = (my_data/np.sum(my_data,0)).transpose()/(np.sum(my_data, 1)/np.sum(my_data))

    print(colsum)

    if switch == 1:
        method = 'PCA'
    else:
        method = 't-SNE'

    ### Plotting fold enrichment

    fig, axes = plt.subplots(figsize=(20,16))
    make_nice_axis(axes)

    colsum2 = colsum*100/np.sum(colsum)

    grid = np.linspace(1.0, np.max(colsum), num=100)

    x = np.linspace(1.0, np.max(colsum2), num=100)

    for enrichment_of_protein in range(1,len(label_legends)+1):
        plt.plot(colsum2, enrichment[:, enrichment_of_protein-1], 'o', label=label_legends[enrichment_of_protein-1], color=colors[enrichment_of_protein-1], markersize = 15)

    plt.plot(grid,np.ones((len(grid),1)),'k-')

    # Labels Cutoff at 99.9% confidence interval
    for i in set(labels):
        if i>=0:
            frac[label_legends[i]] = float(label_counts[i])/sum(label_counts)
            plt.plot(x,1+3.29*np.sqrt(np.divide((1.0-frac[label_legends[i]]),(frac[label_legends[i]]*grid))),colors[i])
            plt.plot(x,1-3.29*np.sqrt(np.divide((1.0-frac[label_legends[i]]),(frac[label_legends[i]]*grid))),colors[i])

    axes.set_xlabel('Cluster size (as percent of dataset)', fontsize=30)
    axes.set_ylabel('Fold Enrichment', fontsize=30)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.legend(fontsize=25)
    
    if flag_save_files:
    
        file_path = write_file_path + output_filename
        create_dir(write_file_path)
        plt.savefig(file_path + '_' + method + '_' + str(num_components) + '_' + str(num_clusters) + '.svg',format='svg',dpi=600,bbox_inches='tight')
        plt.savefig(file_path + '_' + method + '_' + str(num_components) + '_' + str(num_clusters) + '.png',format='png',dpi=600,bbox_inches='tight')
    
    plt.show()
    plt.close()


###############################################################################
#               PLOT CROSS-BLOCKINESS PATTERNS FROM SEQUENCES                 #
###############################################################################

#        file_path = "results/BSD prots Young lab/"

def plot_patterns(prot_names, fasta_file, Motif_filename, file_path, flag_save_files, coarse_graining, custom_aa_list=False, aa_group=None, aa_group_names=None, flag_single_bsd=1, flag_cross_bsd=0, flag_motifs=1):
    
    
    for sequence in SeqIO.parse(fasta_file,'fasta'):
        
        if sequence.id.split('|')[2].split('_')[0] in prot_names:
            
            title_of_figs = sequence.id.split('|')[2].split('_')[0] + '_IDR_' + sequence.id.split('_')[1]
            fig_path = file_path + '/' + title_of_figs + '/'
            create_dir(fig_path)
            print('{} : This IDR is {} amino acids long'.format(title_of_figs, len(sequence.seq)))
            
            if custom_aa_list:
                my_seq = cs.sequence(str(sequence.seq), custom_aa_list=custom_aa_list, aa_group=aa_group, aa_group_names=aa_group_names)
            else:
                my_seq = cs.sequence(str(sequence.seq))
            
            
            if flag_single_bsd:
                my_seq.smear_sequence(coarse_graining)
                my_seq.get_single_bsd()
                block_size_list = my_seq.bsd_single
                group_counter = 0
                for i in block_size_list:
                    x = i.keys()
                    y = i.values()
                    plt.bar(x, y, width=0.8, align='center', alpha=0.5)
                    plt.xlabel('Size of block')
    #                plt.xticks(labels_bsd)
                    plt.ylabel('Frequency')
                    plt.title('%s: BSD of %s'%(title_of_figs,my_seq.aa_group_names[group_counter]))
                    
                    if flag_save_files: 
                        fig_name_single_bsd = fig_path + 'single_bsd'
                        plt.savefig(fig_name_single_bsd + '_%s.png'%(my_seq.aa_group_names[group_counter]),format='png',dpi=600,bbox_inches='tight')
        
                    plt.show()
                    group_counter += 1
            
            if flag_cross_bsd:
                my_seq.smear_sequence(coarse_graining)
                my_seq.get_pairwise_bsd()
                block_size_list = my_seq.bsd_double
                for i in block_size_list:
                    for j in block_size_list[i]:
                        plt.hist(block_size_list[i][j], 10, facecolor='blue', alpha=0.5)
                        plt.xlabel('Size of block')
#                        plt.set_xticks(np.arrange(len(labels_bsd)))
#                        plt.set_xticklabels(labels_bsd)
                        plt.ylabel('Frequency')
                        plt.title('%s: cross BSD of %s and %s'%(title_of_figs,my_seq.aa_group_names[i],my_seq.aa_group_names[j]))
                        
                        if flag_save_files: 
                            fig_name_cross_bsd = fig_path + 'cross_bsd'
                            plt.savefig(fig_name_cross_bsd + '_%s_%s.png'%(my_seq.aa_group_names[i],my_seq.aa_group_names[j]),format='png',dpi=600,bbox_inches='tight')
            
                        plt.show()
                
            if flag_motifs:
                my_seq.read_motifs(Motif_filename)
                my_seq.count_motifs()
                motifs = dict()
                
                for i in range(0,len(my_seq.motif_name_list)):
                    motifs[my_seq.motif_name_list[i]] = my_seq.motif_count[i]
        
                sorted_dict = sorted(motifs.items(), key=itemgetter(1))
                sorted_values = [sorted_dict[i][1] for i in range(0, len(sorted_dict)) if sorted_dict[i][1] != 0]
                sorted_values = sorted_values[::-1]
                sorted_keys = [sorted_dict[i][0] for i in range(0, len(sorted_dict)) if sorted_dict[i][1] != 0]
                sorted_keys = sorted_keys[::-1]
                
                plt.bar(sorted_keys, sorted_values, align='center', facecolor='blue', alpha=0.5)
                plt.xticks(np.arange(len(sorted_keys)), sorted_keys, fontsize=10, rotation='vertical')
                plt.ylabel('Motif count')
                plt.title('%s: Motif counts'%(title_of_figs))
                
                if flag_save_files: 
                    fig_name_motifs = fig_path + 'motif_counts'
                    plt.savefig(fig_name_motifs + '.png',format='png',dpi=600,bbox_inches='tight')
    
                plt.show()
    
# Function to calculate KL divergence between 2 probability distribution functions of 2 AA seqs:

def KL_divergence_single_BSD(fasta_file, prot_id_1, prot_id_2, coarse_graining, custom_aa_list=False, aa_group=False, aa_group_names=False):
    
    from scipy import stats
    
    KL = pd.DataFrame(index = [prot_id_2], columns = aa_group_names)
    
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        
        if seq.id == prot_id_1:
            
            ref_prot = []
            
            if custom_aa_list:
                ref_prot = cs.sequence(str(seq.seq), custom_aa_list=custom_aa_list, aa_group=aa_group, aa_group_names=aa_group_names)
            else:
                ref_prot = cs.sequence(str(seq.seq))
            
            ref_prot.smear_sequence(coarse_graining)
            ref_prot.get_single_bsd()
            block_size_list_ref = ref_prot.bsd_single
        
            
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        
        if seq.id in prot_id_2:
            
            comp_prot = []
            
            if custom_aa_list:
                comp_prot = cs.sequence(str(seq.seq), custom_aa_list=custom_aa_list, aa_group=aa_group, aa_group_names=aa_group_names)
            else:
                comp_prot = cs.sequence(str(seq.seq))
            
            comp_prot.smear_sequence(coarse_graining)
            comp_prot.get_single_bsd()
            block_size_list_comp = comp_prot.bsd_single
        
            for i in range(0,len(block_size_list_comp)):
                
                if len(list(block_size_list_comp[i])) != 0:
                    min_comp = min(list(block_size_list_comp[i]))
                    max_comp = max(list(block_size_list_comp[i]))
                else:
                    min_comp = 0
                    max_comp = 0
                
                if len(list(block_size_list_ref[i])) != 0:
                    min_ref = min(list(block_size_list_ref[i]))
                    max_ref = max(list(block_size_list_ref[i]))
                else:
                    min_ref = 0
                    max_ref = 0
                    
                if min_ref<=min_comp:
                    min_x = min_ref
                else:
                    min_x = min_comp
                
                if max_ref>= max_comp:
                    max_x = max_ref
                else:
                    max_x = max_comp
                    
                x = list(set(list(block_size_list_ref[i].keys()) + list(block_size_list_comp[i].keys()) + list(range(min_x, max_x + 1))))
                
                ref = []
                comp = []
                
                counter=0
                kl_value = 0
                for bl_sz in x:
                    if bl_sz in list(block_size_list_ref[i]):
                        ref.append(block_size_list_ref[i][bl_sz])
                    else:
                        ref.append(0)
                        
                    if bl_sz in list(block_size_list_comp[i]):
                        comp.append(block_size_list_comp[i][bl_sz])
                    else:
                        comp.append(0)
                    
                    if comp[counter] != 0:
                        kl_value += ref[counter]*np.log10(ref[counter]/comp[counter])
                    else:
                        kl_value += ref[counter]*np.log10(ref[counter])
                    
                KL.iloc[np.where(KL.index==seq.id)[0], np.where(KL.columns==aa_group_names[i])[0]] = kl_value
#            ref = 
#            comp = 
                
            
        #KL[seq.id] = stats.entropy(t1.pdf(x), t2.pdf(x))
        
        
    
        