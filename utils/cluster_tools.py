#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:32:26 2018

This function provides a list of clustering methodologies
@author: krishna
"""

import numpy
from sklearn import manifold
from scipy.cluster.hierarchy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import matplotlib; mpl =matplotlib
mpl.rcParams['figure.figsize'] = (12.0,9.0) # default = (6.0, 4.0)
mpl.rcParams['font.size']      = 28        # default = 10

mpl.rcParams['axes.linewidth']    = 0.75 # default = 1.0
mpl.rcParams['lines.linewidth']   = 1.5 # default = 1.0
mpl.rcParams['patch.linewidth']   = 1.0 # default = 1.0
mpl.rcParams['grid.linewidth']    = 0.5 # default = 0.5
mpl.rcParams['xtick.major.width'] = 1.0 # default = 0.5
mpl.rcParams['xtick.minor.width'] = 0.0 # default = 0.5
mpl.rcParams['ytick.major.width'] = 1.0 # default = 0.5
mpl.rcParams['ytick.minor.width'] = 0.0 # default = 0.5


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
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 20


def PCA_analysis(obs_matrix,centered=True,clean_matrix=False,clean_threshold=2.0,write_PC_scores=False,write_path=''):

    """
        Pass the observation matrix in the form
        (nsamples,nfeatures) with an optional flag of mean centering
        + std normalization (default is True).

        Function calculates pc_scores, eig_vals, and eig_vecs
    """
    if centered:
        mean_vec = numpy.mean(obs_matrix, axis=0)
        q_cent = (obs_matrix - mean_vec)/numpy.std(obs_matrix, axis=0)
    else:
        q_cent = obs_matrix

    #  If centering occurs, the calculated matrix is the correlation matrix

    cov_mat = (q_cent).T.dot(q_cent) / (q_cent.shape[0]-1)
    eig_vals, eig_vecs = numpy.linalg.eig(cov_mat)

    key = numpy.argsort(eig_vals)[::-1]
    eig_vals, eig_vecs = eig_vals[key], eig_vecs[:, key]

    if clean_matrix:
        q = obs_matrix.shape[0]/(obs_matrix.shape[1])
        lambda_max = 1 + 1/q + 2/q**0.5

        pos = numpy.where(eig_vals < clean_threshold*lambda_max)[0][0]
        cleaned_cov = (eig_vecs[:, :pos].dot(numpy.diag(eig_vals[:pos]))).dot(eig_vecs[:, :pos].T)
        eig_vals, eig_vecs = numpy.linalg.eig(cleaned_cov)

        key = numpy.argsort(eig_vals)[::-1]
        eig_vals, eig_vecs = eig_vals[key], eig_vecs[:, key]

    pc_scores = (q_cent.dot(abs(eig_vecs))).astype(float)

    if write_PC_scores:

        col_labels = ['pc_'+str(x+1) for x in range(len(eig_vals))]
        df = pd.DataFrame(pc_scores, columns=col_labels)
        os.makedirs(write_path, exist_ok=True)
        file_name = write_path + '/pca_clean_' + str(clean_matrix) + '.csv'
        df.to_csv(file_name)

    return eig_vals, abs(eig_vecs), pc_scores


def tSNE_projection(obs_matrix,n_components=2,perplexity=50,centered=True,write_tSNE_scores=False,write_path=''):

    """
        Takes in the obs_matrix and returns
        the tSNE projection matrix Y
    """
    if centered:
        mean_vec = numpy.mean(obs_matrix, axis=0)
        q_cent = (obs_matrix -mean_vec)/numpy.std(obs_matrix,axis=0)
    else:
        q_cent = obs_matrix

    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(q_cent)

    if write_tSNE_scores:

        col_labels = ['tNSE_'+str(x+1) for x in range(n_components)]
        df = pd.DataFrame(Y,columns=col_labels)
        os.makedirs(write_path,exist_ok=True)
        file_name = write_path + '/tSNE_ncomp_' + str(n_components) + '_perp_' + str(perplexity) + '.csv'
        df.to_csv(file_name)

    return Y

def hierarchical_cluster(obs_matrix,method='ward',optimal_ordering=True,n_clusters=2):

    """
        Z calculates the clustered tree, and T the identified labels of each
        species for splitting tree into n_clusters. Returns Z,T
    """

    Z = linkage(obs_matrix, method = method,optimal_ordering=optimal_ordering)
    T = fcluster(Z,n_clusters, 'maxclust')
    return (Z,T)


def plot_PCA(pc_scores,T=None,pc_int1=0,pc_int2=1,colors=None,labels=None,file_save=True,write_path=''):

    """
        Pass the pc_scores matrix, labels for various points in
        array T, as well as optional PC axis to plot along. Defaults
        are PC1 and PC2.
    """

    figure_pca,ax = plt.subplots()
    make_nice_axis(ax)

    ax.set_title('PCA analysis');

    if colors is None:
        colors = ['grey'];
    if labels is None:
        labels = ['data'];

    if T is None:
        T=[1];
        ax.scatter(pc_scores[count,pc_int1],pc_scores[count,pc_int2],color=colors[0],label=labels[0])
    else:
        labs = list(set(T));
        count = 0;
        for label in labs:
            rel_indices = numpy.where(T==label)[0];
            ax.scatter(pc_scores[rel_indices,pc_int1],pc_scores[rel_indices,pc_int2],color=colors[count],label=labels[count])
            count +=1;
    ax.legend(bbox_to_anchor=(1.05,1.1))

    plt.xlabel('PC'+ str(pc_int1+1))
    plt.ylabel('PC'+ str(pc_int2+1))

    if file_save:
        write_file = write_path + '/pc_' +str(pc_int1) + '_vs_pc_' + str(pc_int2)
        os.makedirs(write_path,exist_ok=True)
        figure_pca.savefig(write_file+'.svg',format='svg',dpi=300,bbox_inches='tight')
        figure_pca.savefig(write_file+'.png',format='png',dpi=300,bbox_inches='tight')

    return ax


def plot_PC(eig_vecs,labels,pc_int, figsize=(6,4)):


    """
        Pass the eigen vector column matrix, labels for xpos,
        as well as PC of interest? Figure size is an optional argument
    """
    random_vec = numpy.ones(len(labels))/pow(len(labels),0.5)
    random_vec_2 = -1*numpy.ones(len(labels))/pow(len(labels),0.5)

    f,ax = plt.subplots(figsize=figsize)
    x= numpy.arange(len(labels))
    ax.bar(x,eig_vecs[x,pc_int], align='center', alpha=0.5)
    ax.plot(x,random_vec,'r--')
    ax.plot(x,random_vec_2,'r--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Relative weight of eigen vector')
    ax.set_title('How do different amino acids matter along PC ' + str(pc_int+1) + '?')
    return ax;

def plot_tSNE(Y,T,ax1=0,ax2=1,colors=None,labels=None,file_save=False,write_path=''):

    """
        Pass the tSNE projection matrix, labels for the various points in
        each row of Y, and optional axis to plot along ax1, ax2.

        By default ax1=0, ax2=1
    """
    figure_tsne,ax = plt.subplots();
    make_nice_axis(ax)

    if colors is None:
        colors = ['grey'];
    if labels is None:
        labels = ['data'];

    if T is None:
        T=[1];
        ax.scatter(Y[count,ax1],Y[count,ax2],color=colors[0],label=labels[0])
    else:
        labs = list(set(T));
        count = 0;
        for label in labs:
            rel_indices = numpy.where(T==label)[0];
            ax.scatter(Y[rel_indices,ax1],Y[rel_indices,ax2],color=colors[count],label=labels[count])
            count +=1;

    ax.axis('tight')
    ax.set_xlabel('tSNE axis ' +str(ax1+1))
    ax.set_ylabel('tSNE axis ' +str(ax2+1))

    ax.legend(bbox_to_anchor=(1.05,1.1))

    if file_save:
        write_file = write_path + '/tSNE_' +str(ax1) + '_vs_tSNE_' + str(ax2)
        os.makedirs(write_path,exist_ok=True)
        figure_tsne.savefig(write_file+'.svg',format='svg',dpi=300,bbox_inches='tight')
        figure_tsne.savefig(write_file+'.png',format='png',dpi=300,bbox_inches='tight')


    return ax;
