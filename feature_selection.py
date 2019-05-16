import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


gapped = ""  # "without_gapped_" or ""
experiment = 'cv'  # 'cv' or 'cv_without_gapped'

dimensions = []
for i in range(1, 6):
    train_x = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'all_' + str(i) + '.h5'))
    train_y = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'all_' + str(i) + '.h5')).iloc[:, 0]

    extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1, verbose=1)
    sfm = SelectFromModel(estimator=extraTree)
    sfm.fit(train_x, train_y)
    train_x = sfm.transform(train_x)
    dimensions.append(train_x.shape[1])

pd.Series(dimensions).to_csv('Results/dimensions_' + experiment + '.csv', index=False, header=False)
