import pandas as pd
from sklearn.svm import SVC
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier


i = 1

cells = ['hct116', 'hek293', 'hela', 'hl60']

gapped = 'without_gapped_'  # 'without_gapped_', ''

for leave_cell in cells:
    train_cells = list(set(cells) - {leave_cell})
    train_x = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + train_cells[0] + '_' + str(i) + '.h5',
                                       key=train_cells[0]))
    train_y = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + train_cells[0] + '_' + str(i) + '.h5',
                                       key=train_cells[0]))
    for j in range(1, 3):
        train_x = train_x.append(pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped +
                                              train_cells[j] + '_' + str(i) + '.h5',
                                                          key=train_cells[j]))).reset_index(drop=True)
        train_y = train_y.append(pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped +
                                              train_cells[j] + '_' + str(i) + '.h5',
                                                          key=train_cells[j]))).reset_index(drop=True)

    extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1)

    steps = [('SFM', SelectFromModel(estimator=extraTree)),
             ('scaler', StandardScaler()),
             ('SVM', SVC(C=10, gamma=0.001, kernel='rbf',
                         random_state=1, probability=True, cache_size=20000, verbose=2))]

    pipeline = Pipeline(steps)

    pipeline.fit(train_x, train_y)

    f = open('Saved Models/leave_' + leave_cell + '_trained_crisprpred_seq_' + gapped + str(i) + '.pkl', 'wb')
    pkl.dump(pipeline, f)
