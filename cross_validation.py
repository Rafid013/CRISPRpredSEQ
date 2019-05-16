import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import pickle as pkl

i = 3

gapped = "without_gapped_"  # "without_gapped_" or ""
experiment = 'cv_without_gapped'  # 'cv' or 'cv_without_gapped'

train_x = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'all_' + str(i) + '.h5'))
train_y = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'all_' + str(i) + '.h5')).iloc[:, 0]

cells = ['ALL', 'HCT116', 'HEK293', 'HELA', 'HL60']

f_log = open('Logs/cross_validation_log_' + gapped + str(i) + '.txt', 'w')

extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1)

steps = [('SFM', SelectFromModel(estimator=extraTree)),
         ('scaler', StandardScaler()),
         ('SVM', SVC(kernel='rbf', random_state=1, probability=True, cache_size=20000, verbose=2, shrinking=False))]

pipeline = Pipeline(steps)

parameters = {'SVM__C': [1, 10, 100],
              'SVM__gamma': [0.0001, 0.001, 0.01]}

scoring = {'roc_auc': 'roc_auc',
           'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall'}

grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1,
                    pre_dispatch=4, scoring=scoring,
                    refit='roc_auc', verbose=2)

grid.fit(train_x, train_y)

f_log.write('FOLD ' + str(i) + '\n')
f_log.write("Best Score: " + str(grid.best_score_) + "\n")
f_log.write("Best Parameters: " + str(grid.best_params_) + "\n\n\n\n")
mean_rocs = grid.cv_results_['mean_test_roc_auc']
std_rocs = grid.cv_results_['std_test_roc_auc']
mean_accs = grid.cv_results_['mean_test_accuracy']
std_accs = grid.cv_results_['std_test_accuracy']
mean_pres = grid.cv_results_['mean_test_precision']
std_pres = grid.cv_results_['std_test_precision']
mean_recs = grid.cv_results_['mean_test_recall']
std_recs = grid.cv_results_['std_test_recall']
params = grid.cv_results_['params']
for mean_roc, std_roc, mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, param \
        in zip(mean_rocs, std_rocs, mean_accs, std_accs, mean_pres, std_pres, mean_recs, std_recs, params):
    f_log.write("Mean ROC_AUC: " + str(mean_roc) + "\n")
    f_log.write("Standard Deviation ROC_AUC: " + str(std_roc) + "\n")
    f_log.write("Mean ACCURACY: " + str(mean_acc) + "\n")
    f_log.write("Standard Deviation ACCURACY: " + str(std_acc) + "\n")
    f_log.write("Mean PRECISION: " + str(mean_pre) + "\n")
    f_log.write("Standard Deviation PRECISION: " + str(std_pre) + "\n")
    f_log.write("Mean RECALL: " + str(mean_rec) + "\n")
    f_log.write("Standard Deviation RECALL: " + str(std_rec) + "\n")
    f_log.write("Parameters: " + str(param) + "\n")
    f_log.write('.........\n\n\n')
f_log.close()

f = open('Saved Models/trained_' + experiment + '_' + str(i) + '.pkl', 'wb')
pkl.dump(grid, f)
