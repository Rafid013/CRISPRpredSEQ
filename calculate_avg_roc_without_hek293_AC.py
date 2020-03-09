from sklearn.metrics import roc_auc_score
import pickle as pkl
import numpy as np


exp_type = input("Which experiment? A, B or C\n")
exp_type = exp_type.upper()
if exp_type == 'A':
    experiment = 'crisprpred'
    gapped = 'without_gapped_'
elif exp_type == 'B':
    experiment = 'cv_without_gapped'
    gapped = 'without_gapped_'
else:
    experiment = 'cv'
    gapped = ''

cells = ['ALL', 'HCT116', 'HEK293', 'HELA', 'HL60']

for cell in cells:
    f1 = open('Predicted/without_hek293_' + experiment + '_predicted_' + cell.lower() + '.pkl', 'rb')
    f2 = open('Predicted/without_hek293_' + experiment + '_predicted_proba_' + cell.lower() + '.pkl', 'rb')
    predicted = pkl.load(f1)
    predicted_proba = pkl.load(f2)
    predicted_proba = predicted_proba[:, 1].tolist()
    f3 = open('Predicted/without_hek293_' + experiment + '_test_y_' + gapped + cell.lower() + '.pkl', 'rb')
    test_y = pkl.load(f3)

    print(cell)
    roc = roc_auc_score(test_y, predicted_proba)
    print("ROC_AUC: " + str(roc))
    print('\n\n')
