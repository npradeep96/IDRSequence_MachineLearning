import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#ref for code
# https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

# load the normalized data
df = pd.read_pickle('normData.pkl')

shp = np.shape(df)
N =  int(shp[0]) # number of data points
nF = int(shp[1]) # number of features

# separate out target from features
x = df.iloc[:,0:-1] # all but the last column are the features 
y = df.iloc[:,-1] # last column is the labels

#standardize the data
x = StandardScaler().fit_transform(x)

pca = PCA().fit(x)

# matplotlib inline
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, nF, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, nF, step=20)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()



#pca = PCA(.95) # choose number of components to retain 95% of variance
#principalComponents = pca.fit_transform(x)
#print(np.shape(principalComponents))

#pca = PCA(n_components = 2)
#principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])

#finalDf = pd.concat([principalDf, y], axis = 1)

# plot if you want
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component PCA', fontsize = 20)
#targets = [0.0, 1.0]
#colors = ['r', 'g']
#for target, color in zip(targets,colors):
#    indicesToKeep = finalDf.iloc[:,-1] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , c = color
#               , s = 50)
#ax.legend(targets)
#ax.grid()

#plt.show()
