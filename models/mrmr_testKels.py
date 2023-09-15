import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#import seaborn as sns
import pymrmr

# file to play around w mrmr for features 


#how many features to select
numF = list(np.linspace(10, 60, 6))
fileName = 'mrmr_10to60_alpha0.5_MID_ThursDec8.pkl'



# load the normalized data
df = pd.read_pickle('normData.pkl')

shp = np.shape(df)
N =  int(shp[0]) # number of data points
nF = int(shp[1]) # number of features

# find the column number where the short linear motif data starts
indLM = df.columns.get_loc('MOD_CDK_SPK_2')

# take out the category that has the class labels
# for mrmr package, this needs to be the first column.
labels = df[df.columns[nF-1]]
# ignore the short linear motifs
#df.drop(df.columns[144:227],axis=1,inplace=True)
df.drop(df.columns[(nF-1):nF],axis=1,inplace=True)
# list of column names
varName = list(df)

# discretize the features according to mean +/- alpha*std
# first get the mean and std dev of each feature column
mean = df.mean()
std = df.std()
#now categorize them
a = 0.5
for i in range(0,(nF-1)):
     bins=[0.0 , (mean[i] - a*std[i]), (mean[i] + a*std[i]), 1.0]
     print(varName[i])
     if bins[1]<0:
           bins[1] = 0.00001
     pd.cut(df.iloc[:, i], bins=bins) #go from continuous to categorical

#add the labels as the first column of pd
df.insert(0,'labels', labels)

# run mrmr
fIndices = []
for m in range(0,len(numF)):
     fIndices_m = pymrmr.mRMR(df, 'MID', numF[m])
     fIndices.append(fIndices_m)     

with open(fileName , 'wb') as f:
     pickle.dump(fIndices,f)
