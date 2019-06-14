import pandas as pd
from sklearn.svm import SVC
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np


for i in range(1, 6):
    train_x = pd.DataFrame(pd.read_hdf('Folds/train_x_without_gapped_all_' + str(i) + '.h5'))
    train_y = pd.DataFrame(pd.read_hdf('Folds/train_y_without_gapped_all_' + str(i) + '.h5')).iloc[:, 0]

    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=1, verbose=2)

    steps = [('SFM', SelectFromModel(estimator=rf, max_features=2899, threshold=-np.inf)),
             ('scaler', StandardScaler()),
             ('SVM', SVC(C=1, gamma='auto', kernel='rbf',
                         random_state=1, probability=True, cache_size=20000, verbose=2))]

    pipeline = Pipeline(steps)

    pipeline.fit(train_x, train_y)

    f = open('trained_crisprpred_' + str(i) + '.pkl', 'wb')
    pkl.dump(pipeline, f)
