# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:48:20 2019

@author: chemegrad2018
"""

import numpy as np
from collections import Counter
import re

class sequence:

    ############### Static variables to share across instances ################

    aa_list = ['E','D', 'R','K','H', 'Q','N','S','T','G', 'C', 'A','L','M','I','V','F','Y','W','P']

    aa_groups = [['I','L','V','F','W','Y','M','G','A'], ['E','D'], ['R','K','H'], ['E','D','Q','N','R','H','F','Y','W'],['S','T','Q','N','C'],['P']]
    aa_group_names = ['hydrophobic','neg_charge','pos_charge','sidechain_pipi','polar','proline']

    num_groups = len(aa_groups)

    num_features_from_bsd = 7
    names_features_from_bsd = ['P_small', 'P_medium', 'P_large', 'l_avg', 'l_avg_norm', 'l_max', 'l_max_norm']


    ############## Class constructor ##########################################

    def __init__(self, input_seq = '',custom_aa_list=False,aa_group=None,aa_group_names=None):

        self.seq = input_seq
        if custom_aa_list:
            self.aa_groups = aa_group;
            self.aa_group_names = aa_group_names;
            self.num_groups = len(aa_group)

    ############ String representation of the class ###########################

    def __str__(self):

        return str(vars(self))

    ######################### Function to smear out sequences #################

    def smear_sequence(self, coarse_graining, window_length = 6):

        self.smeared_seq = {}

        for k in range(0,self.num_groups):

            self.smeared_seq[self.aa_group_names[k]] = ''

            for i in range(0,len(self.seq) - window_length):

#                if set(self.seq[i:(i + window_length)]).intersection(self.aa_groups[k]) == set():
#                    self.smeared_seq[self.aa_group_names[k]] = self.smeared_seq[self.aa_group_names[k]] + '0'
                if len([s for s in list(self.seq[i:(i + window_length)]) if s in (self.aa_groups[k])])>= coarse_graining:
                    self.smeared_seq[self.aa_group_names[k]] = self.smeared_seq[self.aa_group_names[k]] + '1'
                else:
                    self.smeared_seq[self.aa_group_names[k]] = self.smeared_seq[self.aa_group_names[k]] + '0'

   ########## Function to find the coarse-grained group composition ##########

    def get_composition(self):

        self.composition = np.zeros(self.num_groups)

        counts = Counter(self.seq)

        for i in range(0,self.num_groups):

            for j in range(0,len(self.aa_groups[i])):

                self.composition[i] = self.composition[i] + counts[self.aa_groups[i][j]]

            self.composition[i] = self.composition[i]/len(self.seq)

        self.composition = list(self.composition)

    ####### Function to calculate single block size distribution ####

    def get_single_bsd(self):

        self.bsd_single = []

        for gp in self.aa_group_names:

            bs_list = []

            start_idx = self.smeared_seq[gp].find('1')
            end_idx = 0

            while start_idx >= 0:

                start_idx += end_idx
                end_idx = self.smeared_seq[gp][start_idx:].find('0')

                if end_idx >= 0:
                    bs_list.append(end_idx)
                    end_idx += start_idx
                    start_idx = self.smeared_seq[gp][end_idx:].find('1')
                else:
                    bs_list.append(len(self.smeared_seq[gp]) - start_idx)
                    break

            temp_bsd_single =  Counter(bs_list)
            no_of_blocks = sum(temp_bsd_single.values())
            for i in temp_bsd_single.keys():
                temp_bsd_single[i] = float(temp_bsd_single[i])/float(no_of_blocks)

            self.bsd_single.append(temp_bsd_single)


    ####### Function to get pairwise block size distributions ###########

    def get_pairwise_bsd(self):

        self.bsd_double = []
        N = len(self.smeared_seq[self.aa_group_names[0]])

        for gp1 in self.aa_group_names:

            temp_bsd_double_D1 = []

            for gp2 in self.aa_group_names:

                bs_list = []

                for idx in range(0,N):

                    if self.smeared_seq[gp1][idx] == '1':

                        bs_1 = self.smeared_seq[gp2][(idx+1):].find('0')
                        if bs_1 >= 0 :
                            bs_list.append(bs_1)
                        else:
                            bs_list.append(N-idx-1)

                        bs_2 = self.smeared_seq[gp2][:idx][::-1].find('0')
                        if bs_2 >= 0:
                            bs_list.append(bs_2)
                        else:
                            bs_list.append(idx)


                temp_bsd_double_D2 = Counter(bs_list)
                no_of_blocks = sum(temp_bsd_double_D2.values())
                for i in temp_bsd_double_D2.keys():
                    temp_bsd_double_D2[i] = float(temp_bsd_double_D2[i])/float(no_of_blocks)

                temp_bsd_double_D1.append(temp_bsd_double_D2)

            self.bsd_double.append(temp_bsd_double_D1)

    ############ Function to count the number of occurences of a motif from a list

    def read_motifs(self, Motif_filename):

        """
        In this function, I have assumed that the motifs are stored in a file where each line is of the form:

        motif-name, motif-regex
        """

        f = open(Motif_filename, "r")
        motifs_from_file = list(f.readlines())
        motifs_from_file = [x.strip('\n') for x in motifs_from_file]
        motifs_from_file = [x.strip('\r') for x in motifs_from_file]
        f.close()

        self.motif_name_list = [x.split(', ')[0] for x in motifs_from_file]
        self.motif_regex_list = [x.split(', ')[1] for x in motifs_from_file]

    def count_motifs(self):

        self.motif_count = list()
        L = len(self.motif_regex_list)

        for i in range(0,L):

            self.motif_count.append( len(re.findall( str(self.motif_regex_list[i]), str(self.seq) )) )


    #### Function to extract the above seven features from the block size distribution

    def get_features_from_bsd(self):

        self.features_from_bsd = list()

        for idx1 in range(0, self.num_groups):

            self.features_from_bsd.append( self.calculate_features_for_given_bsd(self.bsd_single[idx1]) )

        for idx1 in range(0, self.num_groups):
            for idx2 in range(0, self.num_groups):
                self.features_from_bsd.append( self.calculate_features_for_given_bsd(self.bsd_double[idx1][idx2]) )


    def calculate_features_for_given_bsd(self, bsd):

        self.num_features_from_bsd = 7
        features_to_return = self.num_features_from_bsd * [0]

        features_to_return[0] = self.get_probability_of_block_size( bsd , -0.01, 1.0/3.0 )
        features_to_return[1] = self.get_probability_of_block_size( bsd , 1.0/3.0, 2.0/3.0 )
        features_to_return[2] = self.get_probability_of_block_size( bsd , 2.0/3.0, 1.0 )

        features_to_return[3] = self.get_avg_of_block_size( bsd )
        try:
            features_to_return[4] = features_to_return[3] / len(self.smeared_seq[self.aa_group_names[0]])
        except ZeroDivisionError:
            features_to_return[4] = 0.0

        if bsd.keys():
            features_to_return[5] = max(bsd.keys())
            try:
                features_to_return[6] = float( features_to_return[5] ) / float( len(self.smeared_seq[self.aa_group_names[0]]) )
            except ZeroDivisionError:
                features_to_return[6] = 0.0


        return features_to_return


    def get_probability_of_block_size(self, bsd, fmin, fmax):

        if bsd.keys():
            max_block_size = max(bsd.keys())

        prob = 0

        for idx1 in bsd.keys():

            if idx1 <= fmax*max_block_size and idx1 > fmin*max_block_size:

                prob += bsd[idx1]

        return prob

    def get_avg_of_block_size(self, bsd):

        avg_of_block_size = 0

        for idx1 in bsd.keys():

            avg_of_block_size += idx1*bsd[idx1]

        return avg_of_block_size
