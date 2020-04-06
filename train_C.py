import pandas as pd
from sklearn.svm import SVC
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import sys
import os


filepath = sys.argv[1]
filename = os.path.splitext(os.path.basename(filepath))[0]


pi = pd.DataFrame(pd.read_hdf('Data/' + filename + '_pi.h5', key='pi')).astype(np.int8)
ps = pd.DataFrame(pd.read_hdf('Data/' + filename + '_ps.h5', key='ps')).astype(np.int8)
gap = pd.DataFrame(pd.read_hdf('Data/' + filename + '_gap.h5', key='gap')).astype(np.int8)
labels = pd.DataFrame(pd.read_hdf('Data/' + filename + '_labels.h5', key='labels')).astype(np.int8)

features = pd.concat([pi, ps, gap], axis=1, sort=False).astype(np.int8)

extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1, verbose=2)

steps = [('SFM', SelectFromModel(estimator=extraTree)),
         ('scaler', StandardScaler()),
         ('SVM', SVC(C=10, gamma=0.001, kernel='rbf', random_state=1, probability=True, cache_size=20000, verbose=2,
                     shrinking=False))]

pipeline = Pipeline(steps)

pipeline.fit(features, labels)

f = open('Saved Models/' + filename + '_C.pkl', 'wb')
pkl.dump(pipeline, f)
