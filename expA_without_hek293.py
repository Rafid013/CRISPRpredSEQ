import pandas as pd
from sklearn.svm import SVC
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np


train_x = pd.DataFrame(pd.read_hdf('Folds/train_without_hek293_x_without_gapped.h5'))
train_y = pd.DataFrame(pd.read_hdf('Folds/train_without_hek293_y_without_gapped.h5')).iloc[:, 0]

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=1, verbose=3)

steps = [('SFM', SelectFromModel(estimator=rf, max_features=2899, threshold=-np.inf)),
         ('scaler', StandardScaler()),
         ('SVM', SVC(C=1, gamma='auto', kernel='rbf',
                     random_state=1, probability=True, cache_size=20000, verbose=2))]

pipeline = Pipeline(steps)

pipeline.fit(train_x, train_y)

f = open('Saved Models/trained_without_hek293_crisprpred.pkl', 'wb')
pkl.dump(pipeline, f)
