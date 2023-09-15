"""
Module of functions to:
- Process and filter output of IDR/PFAM data from D2P2 database
- Characterize distributions in IDR/PFAM data
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def exclude_idrs_pfams(df):
    """
    Input is the data-frame of the d2p2 data.

    Modifies the data-frame directly to remove IDR sequences that
    lie completely within pfam domains.

    Prints the number of IDRs removed

    """

    idr_change = 0;
    for seq in range(df.shape[0]):
        idr_ids = df['IDR_id'][seq].split(',');
        idr_start = [int(x) for x in df['idr_start'][seq].split(',')];
        idr_end = [int(x) for x in df['idr_end'][seq].split(',')];

        if (df['pfam_all'][seq] != 'None' and df['pfam_all'][seq] != 'None_') and (idr_start[0]>0):

            pfam_domain = df['pfam_all'][seq].split('_,')
            pfam_start = [int(x.split('_')[1]) for x in pfam_domain];
            pfam_end = [int(x.split('_')[2]) for x in pfam_domain];

            flag =0;
            while(flag>=0):

                inside_pfam = [(idr_start[flag]>= pfam_start[x] and idr_end[flag] <=pfam_end[x]) for x in range(len(pfam_domain))];
                if True in inside_pfam:
                    del idr_ids[flag],idr_start[flag],idr_end[flag];
                    idr_change+=1;
                    if(flag>len(idr_ids)-1):
                        flag =-1;

                else:
                    if(flag<len(idr_ids)-1):
                        flag +=1;
                    else:
                        flag=-1;

                if not idr_ids:
                    flag=-1;
                    idr_ids = [df['ID'][seq]];
                    idr_start = [-1];
                    idr_end = [-1];


            df['IDR_id'].values[seq] = ','.join(idr_ids);
            df['idr_start'].values[seq] = ','.join(str(x) for x in idr_start);
            df['idr_end'].values[seq] = ','.join(str(x) for x in idr_end);

    print('Total IDRs excluded are {}'.format(idr_change))


def exclude_linker_idrs(df,linker_gap=30):
    """
    Input is the data-frame of the d2p2 data and linker_gap

    Modifies the data-frame directly to remove IDR sequences that
    lie completely between 2 PFAM domains that are less than linker_gap amino
    acids away from each other.

    Prints the number of IDRs removed
    """

    idr_change = 0;

    for seq in range(df.shape[0]):
        idr_ids = df['IDR_id'][seq].split(',');
        idr_start = [int(x) for x in df['idr_start'][seq].split(',')];
        idr_end = [int(x) for x in df['idr_end'][seq].split(',')];

        if (df['pfam_all'][seq] != 'None' and df['pfam_all'][seq] != 'None_') and (idr_start[0]>0):

            pfam_domain = df['pfam_all'][seq].split('_,')
            pfam_start = [int(x.split('_')[1]) for x in pfam_domain];
            pfam_end = [int(x.split('_')[2]) for x in pfam_domain];
            diff = [(pfam_start[idr]-pfam_end[idr-1]) for idr in range(1,len(pfam_domain))];

            if diff:
                list_of_ids = [];
                for idr in range(len(idr_ids)):
                    list_of_ids += list(set([idr for domain in range(len(diff)) if ((idr_start[idr]>pfam_end[domain]) and (idr_end[idr]<pfam_start[domain+1]) and (diff[domain]<=linker_gap))]))

                flag =0;
                for idx in list_of_ids:
                    del idr_ids[idx-flag],idr_start[idx-flag],idr_end[idx-flag];
                    flag = flag+1;
                    idr_change+=1;
                if not idr_ids:
                    idr_ids = [df['ID'][seq]];
                    idr_start = [-1];
                    idr_end = [-1];

            df['IDR_id'].values[seq] = ','.join(idr_ids);
            df['idr_start'].values[seq] = ','.join(str(x) for x in idr_start);
            df['idr_end'].values[seq] = ','.join(str(x) for x in idr_end);

    print('Total linker IDRs are {}'.format(idr_change))

def stitch_idrs(df,length_to_stitch=12):
    """
    Input is the data-frame of the d2p2 data, and length_to_stitch

    Modifies the data-frame directly to stitch two IDR sequences in
    the same protein if they are within length_to_stitch AA of each other

    Prints the number of IDRs stitched together
    """

    total_idrs_stitched = 0;
    for seq in range(df.shape[0]):

        idr_ids = df['IDR_id'][seq].split(',');
        idr_start = [int(x) for x in df['idr_start'][seq].split(',')];
        idr_end = [int(x) for x in df['idr_end'][seq].split(',')];

        if (idr_start[0] > -1) and (len(idr_start)>1):
            diff = [(idr_start[idr]-idr_end[idr-1]) for idr in range(1,len(idr_start))];
            flag = 1;
            while((flag>0)):
                if (diff[flag-1] <=length_to_stitch):
                    idr_ids[flag-1] = idr_ids[flag-1] +'_'+ idr_ids[flag];
                    idr_end[flag-1] = idr_end[flag];

                    del idr_ids[flag], idr_start[flag], idr_end[flag];
                    total_idrs_stitched +=1;

                    if flag < len(diff):
                        diff[flag-1] = diff[flag];
                        del diff[flag];
                    else:

                        flag = -1;

                else:
                    if flag < len(diff):
                        flag +=1;
                    else:
                        flag = -1;

            df['IDR_id'].values[seq] = ','.join(idr_ids);
            df['idr_start'].values[seq] = ','.join(str(x) for x in idr_start);
            df['idr_end'].values[seq] = ','.join(str(x) for x in idr_end);

    print('Total IDRs stitched is {}'.format(total_idrs_stitched))

def write_idr_file(df,output_file):
    """
    Input is the d2p2 data-frame and path (output_file) to write fasta files

    Function writes the IDR sequences of the data-frame into a FASTA format in
    the path output_file.
    """
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    with open(output_file,'w+') as f:
        for idx in range(df.shape[0]):
            ist = [str(o) for o in df['idr_start'][idx].split(',') if o != '-1'];
            iend = [str(o) for o in df['idr_end'][idx].split(',') if o != '-1'];
            for p in range(len(ist)):
                id_to_write = df['ID'][idx] + '_' + str(p+1) + '_' + ist[p] + '_' + iend[p] + '|' + df['gene'][idx] + '|' + df['description'][idx];
                seq_to_write = df['sequence'][idx][int(ist[p])-1:int(iend[p])];
                f.write('>' + id_to_write + '\n' + seq_to_write + '\n')
