import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv

from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

svmAlg = svm.SVC()
gnb = GaussianNB()
bnb = BernoulliNB()
tree = tree.DecisionTreeClassifier()
rnd = RandomForestClassifier()

file = 'C:\Users\Lenovo\Desktop\Parkinson Data\Parkinson Data.csv'
dfwithLabel = DataFrame()
dfwithLabel = read_csv(file, header=None, index_col=None, delimiter=';')

distinct = pd.DataFrame(np.zeros((195, 32), dtype=int))

frames = [dfwithLabel, distinct]

df = DataFrame()
df = pd.concat(frames, axis=1)

numpyMatrix = df.as_matrix()

dist = []

for i in range(0, dfwithLabel.shape[0]):
    illType = dfwithLabel[0][i].split('_')[0] + dfwithLabel[0][i].split('_')[1] + dfwithLabel[0][i].split('_')[2]
    if illType not in dist:
        dist.append(illType)

for y in range(0, numpyMatrix.shape[0]):
    for x in range(0, len(dist)):
        if dist[x] in numpyMatrix[y][0].split('_')[0] + numpyMatrix[y][0].split('_')[1] + numpyMatrix[y][0].split('_')[2]:
            numpyMatrix[y][24+x] = 1

dfwithLabel = pd.DataFrame(data=numpyMatrix)

dfwithLabel = dfwithLabel.drop([0],1)

splitDf = np.split(dfwithLabel, [135], axis=0)

trainDf = splitDf[0]

trainLabel = trainDf[17]
trainDf = trainDf.drop([17],1)
trainDf = trainDf.dropna()
trainLabel = trainLabel.dropna()
trainLabel = trainLabel.apply(int)

testdf = splitDf[1]

testLabel = testdf[17]
testdf = testdf.drop([17],1)
testdf = testdf.dropna()
testLabel = testLabel.dropna()
testLabel = testLabel.apply(int)



try:
    svmModel = svmAlg.fit(trainDf, trainLabel)
    svmpred = svmModel.predict(testdf)
    svmAcc = accuracy_score(testLabel, svmpred)
    print 'SVM Accuracy : ' + str(svmAcc)
    
    gnbModel = gnb.fit(trainDf, trainLabel)
    gnbpred = gnbModel.predict(testdf)
    gnbAcc = accuracy_score(testLabel, gnbpred)
    print 'GNB Accuracy : ' + str(gnbAcc)
    
    bnbModel = bnb.fit(trainDf, trainLabel)
    bnbpred = bnbModel.predict(testdf)
    bnbAcc = accuracy_score(testLabel, bnbpred)
    print 'BNB Accuracy : ' + str(bnbAcc)
    
    treeModel = tree.fit(trainDf, trainLabel)
    treepred = treeModel.predict(testdf)
    treeAcc = accuracy_score(testLabel, treepred)
    print 'Decision Tree Accuracy : ' + str(treeAcc)
    
    rndModel = rnd.fit(trainDf, trainLabel)
    rndpred = rndModel.predict(testdf)
    rndAcc = accuracy_score(testLabel, rndpred)
    print 'Random Forest Accuracy : ' + str(rndAcc)
except Exception, e:
    print 'model error ' + str(e)