import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

file = pd.read_csv('data.csv')

#________________________________________Data Balancing_________________________________________
from collections import Counter
from imblearn.over_sampling import SMOTE

X = file.iloc[:,1:]
Y = file.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30)

SMOTE = SMOTE()

X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

#_______________________________________Feature Selection using chi2________________________________________
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = X_train_SMOTE
Y = y_train_SMOTE
best = SelectKBest(score_func=chi2,k=10)
fit = best.fit(X,Y)

dfscore = pd.DataFrame(fit.scores_)
dfcol = pd.DataFrame(X.columns)

featurescore = pd.concat([dfcol,dfscore], axis = 1)
featurescore.columns = ["feat","Score"]

selectedfeat = featurescore.nlargest(10,"Score")
cols = selectedfeat["feat"]

#______________________________Classification using KNN Classifier (K==1)_______________________________
from sklearn.preprocessing import LabelEncoder
KNC = KNeighborsClassifier(n_neighbors=1)
X_train, X_test, y_train, y_test = train_test_split(X_train_SMOTE[cols], y_train_SMOTE, test_size=0.3, random_state=42)
y_prediction = KNC.fit(X_train, y_train)

pickle.dump(y_prediction, open('model.pkl', 'wb'))
