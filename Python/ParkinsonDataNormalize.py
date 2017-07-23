
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv


# In[2]:

file = 'C:\Users\Lenovo\Desktop\Parkinson Data\Parkinson Data.csv'
df = DataFrame()
df = read_csv(file, header=None, index_col=None, delimiter=';')


# In[3]:

zeroDf = pd.DataFrame(np.zeros((195, 32), dtype=int))

frames = [df, zeroDf]
mergeddf = DataFrame()
mergeddf = pd.concat(frames, axis=1)
numpyMatrix = mergeddf.as_matrix()


# In[5]:

dist = []

for i in range(0, df.shape[0]):
    illType = df[0][i].split('_')[0] + df[0][i].split('_')[1] + df[0][i].split('_')[2]
    if illType not in dist:
        dist.append(illType)

for y in range(0, numpyMatrix.shape[0]):
    for x in range(0, len(dist)):
        if dist[x] in numpyMatrix[y][0].split('_')[0] + numpyMatrix[y][0].split('_')[1] + numpyMatrix[y][0].split('_')[2]:
            numpyMatrix[y][24+x] = 1


# In[7]:

df = pd.DataFrame(data=numpyMatrix)

df = df.drop([0],1)


# In[10]:

df.to_csv('C:\Users\Lenovo\Desktop\Parkinson Data\ParkinsonNormalizedData.csv', sep=',', encoding='utf-8', header=False, index=False)


# In[ ]:



