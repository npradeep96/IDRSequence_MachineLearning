import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# file to play around w features


# load the normalized data
df = pd.read_pickle('normData.pkl')


# ignore the short linear things
df.drop(df.columns[144:277],axis=1,inplace=True)

# get correlation matrix
Cmat = df.corr()

# plot but this is useless w too many features so just ignore
#plt.figure(figsize=(8,8))
_ = sns.heatmap(Cmat)
#mask = np.triu(np.ones_like(Cmat, dtype=bool))
# Create a custom diverging palette
#cmap = sns.diverging_palette(250, 15, s=75, l=40,
#                             n=9, center="light", as_cmap=True)

#_ = sns.heatmap(Cmat, mask = mask, center=0, annot=True, 
#                fmt='.2f', square=True)

#plt.show()

# find highly correlated features
Cmatabs = df.corr().abs()
mask = np.triu(np.ones_like(Cmatabs, dtype=bool))
reduced_matrix = Cmatabs.mask(mask)

# Find columns that meet specified threshold and drop them
thresh =0.95
to_drop = [c for c in reduced_matrix.columns if any(reduced_matrix[c] > thresh)]

print(to_drop)
print(np.shape(to_drop))
