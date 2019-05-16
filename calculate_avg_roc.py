from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle as pkl
import numpy as np


experiment = 'cv_without_gapped'  # 'cv', 'cv_without_gapped', 'crisprpred'
gapped = 'without_gapped_'  # 'without_gapped_', ''

cells = ['ALL', 'HCT116', 'HEK293', 'HELA', 'HL60']

for cell in cells:
    total_predicted = []
    total_predicted_proba = []
    total_y = []
    for i in range(1, 6):
        f1 = open('Predicted/' + experiment + '_predicted_' + cell.lower() + '_' + str(i) + '.pkl', 'rb')
        f2 = open('Predicted/' + experiment + '_predicted_proba_' + cell.lower() + '_' + str(i) + '.pkl', 'rb')
        predicted = pkl.load(f1)
        predicted_proba = pkl.load(f2)
        total_predicted += predicted.tolist()
        total_predicted_proba += predicted_proba[:, 1].tolist()
        if cell == 'ALL':
            key = None
        else:
            key = cell.lower()
        test_y = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                          key=key)).iloc[:, 0]
        total_y += np.array(test_y).tolist()

    print(cell)
    roc = roc_auc_score(total_y, total_predicted_proba)
    print("ROC_AUC: " + str(roc))
    print('\n\n')
