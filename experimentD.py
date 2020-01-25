import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import pickle as pkl

gapped = "without_gapped_"  # "without_gapped_" or ""
experiment = 'D'  # 'D' or 'E'
cells = ['hct116', 'hek293', 'hela', 'hl60']

for i in range(1, 6):
    for leave_out_cell in cells:
        train_x = pd.DataFrame(pd.read_hdf('Leave Folds/train_x_' + gapped + '_leave_' +
                                           leave_out_cell.lower() + '_' + str(i) + '.h5', key='leave'))
        train_y = pd.DataFrame(pd.read_hdf('Leave Folds/train_y_' + gapped + '_leave_' +
                                           leave_out_cell.lower() + '_' + str(i) + '.h5', key='leave')).iloc[:, 0]

        extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1)

        steps = [('SFM', SelectFromModel(estimator=extraTree)),
                 ('scaler', StandardScaler()),
                 ('SVM', SVC(C=10, gamma=0.001, kernel='rbf',
                             random_state=1, probability=True, cache_size=20000, verbose=2))]

        pipeline = Pipeline(steps)

        pipeline.fit(train_x, train_y)

        f = open('Saved Models/trained_' + experiment + '_leave_' + leave_out_cell + '_' + str(i) + '.pkl', 'wb')
        pkl.dump(pipeline, f)
