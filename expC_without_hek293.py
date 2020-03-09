import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import pickle as pkl

gapped = ""  # "without_gapped_" or ""
experiment = 'cv'  # 'cv' or 'cv_without_gapped'

train_x = pd.DataFrame(pd.read_hdf('Folds/train_without_hek293_x.h5'))
train_y = pd.DataFrame(pd.read_hdf('Folds/train_without_hek293_y.h5')).iloc[:, 0]

extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1)

steps = [('SFM', SelectFromModel(estimator=extraTree)),
         ('scaler', StandardScaler()),
         ('SVM', SVC(C=10, gamma=0.001, kernel='rbf',
                     random_state=1, probability=True, cache_size=20000, verbose=2))]

pipeline = Pipeline(steps)

pipeline.fit(train_x, train_y)

f = open('Saved Models/trained_without_hek293_' + experiment + '.pkl', 'wb')
pkl.dump(pipeline, f)
