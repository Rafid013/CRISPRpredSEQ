import pandas as pd
import pickle as pkl
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np


cells = ['hct116', 'hek293', 'hela', 'hl60']

experiment = 'crisprpred_seq'  # 'crisprpred', 'crisprpred_seq', 'crisprpred_seq_without_gapped'
gapped = ''  # 'without_gapped_', ''

for i in range(1, 6):
    for leave_cell in cells:
        print(str(i) + ' ' + leave_cell)
        test_x = pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped +
                                          leave_cell + '_' + str(i) + '.h5', key=leave_cell)).astype(np.float64)
        test_y = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + leave_cell + '_' + str(i) + '.h5', key=leave_cell))

        f = open('Saved Models/leave_' + leave_cell + '_trained_' + experiment + '_' + str(i) + '.pkl', 'rb')
        model = pkl.load(f)

        f_log = open('Logs/test_leave_' + leave_cell + '_' + experiment + '_log_' + str(i) + '.txt', 'w')

        predicted = model.predict(test_x)
        acc = accuracy_score(test_y, predicted)
        pre = precision_score(test_y, predicted)
        rec = recall_score(test_y, predicted)
        predicted_proba = model.predict_proba(test_x)
        roc = roc_auc_score(test_y, predicted_proba[:, 1])
        f_log.write(leave_cell + '\n')
        f_log.write("TEST ROC_AUC: " + str(roc) + "\n")
        f_log.write("TEST ACCURACY: " + str(acc) + "\n")
        f_log.write("TEST PRECISION: " + str(pre) + "\n")
        f_log.write("TEST RECALL: " + str(rec) + "\n")
        f_log.write('.........\n\n\n\n\n\n\n')

        f_predicted = open('Predicted/leave_' + leave_cell + experiment + '_predicted_' + str(i) + '.pkl', 'wb')
        pkl.dump(predicted, f_predicted)

        f_predicted_proba = open('Predicted/leave_' + leave_cell + experiment + '_predicted_proba_' + str(i) + '.pkl',
                                 'wb')
        pkl.dump(predicted_proba, f_predicted_proba)
