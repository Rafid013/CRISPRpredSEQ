import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import pickle as pkl


gapped = "without_gapped_"  # "without_gapped_" or ""
experiment = 'E'  # 'D', 'E' or 'F'
cells = ['hct116', 'hela', 'hl60']

for leave_out_cell in cells:
    train_x = pd.DataFrame(pd.read_hdf('TMP/train_leave_x_' + gapped +
                                       leave_out_cell.lower() + '.h5', key=leave_out_cell))
    train_y = pd.DataFrame(pd.read_hdf('TMP/train_leave_y_' + gapped +
                                       leave_out_cell.lower() + '.h5', key=leave_out_cell)).iloc[:, 0]

    extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1, verbose=3)

    steps = [('SFM', SelectFromModel(estimator=extraTree)),
             ('scaler', StandardScaler()),
             ('SVM', SVC(C=10, gamma=0.001, kernel='rbf',
                         random_state=1, probability=True, cache_size=20000, verbose=2))]

    pipeline = Pipeline(steps)

    pipeline.fit(train_x, train_y)

    f = open('TMP/trained_' + experiment + '_' + leave_out_cell + '.pkl', 'wb')
    pkl.dump(pipeline, f)
