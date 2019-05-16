from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle as pkl
import numpy as np

experiment = 'crisprpred'  # 'crisprpred', 'crisprpred_seq', 'crisprpred_seq_without_gapped'
gapped = ''  # 'without_gapped_', ''

cells = ['HCT116', 'HEK293', 'HELA', 'HL60']

for leave_cell in cells:
    total_predicted = []
    total_predicted_proba = []
    total_y = []
    for i in range(1, 6):
        f1 = open('Predicted/leave_' + leave_cell.lower() + experiment + '_predicted_' + str(i) + '.pkl', 'rb')
        f2 = open('Predicted/leave_' + leave_cell.lower() + experiment + '_predicted_proba_' + str(i) + '.pkl',
                  'rb')
        predicted = pkl.load(f1)
        predicted_proba = pkl.load(f2)
        total_predicted += predicted.tolist()
        total_predicted_proba += predicted_proba[:, 1].tolist()
        if leave_cell == 'ALL':
            key = None
        else:
            key = leave_cell.lower()
        test_y = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + leave_cell.lower() + '_' + str(i) + '.h5',
                                          key=key)).iloc[:, 0]
        total_y += np.array(test_y).tolist()

    print(leave_cell)
    roc = roc_auc_score(total_y, total_predicted_proba)
    print("ROC_AUC: " + str(roc))
    print('\n\n')
