from sklearn.metrics import roc_auc_score
import pickle as pkl
import numpy as np


exp_type = input("Which experiment? D, E or F\n")
exp_type = exp_type.upper()
if exp_type == 'D':
    experiment = 'D'
    gapped = 'without_gapped_'
elif exp_type == 'E':
    experiment = 'E'
    gapped = 'without_gapped_'
else:
    experiment = 'F'
    gapped = ''

cells = ['HCT116', 'HELA', 'HL60']

total_total_predicted = []
total_total_predicted_proba = []
total_total_y = []
for leave_cell in cells:
    f1 = open('Predicted TMP/' + experiment + '_predicted_' + leave_cell.lower() + '.pkl', 'rb')
    f2 = open('Predicted TMP/' + experiment + '_predicted_proba_' + leave_cell.lower() + '.pkl', 'rb')
    predicted = pkl.load(f1)
    predicted_proba = pkl.load(f2)
    predicted_proba = predicted_proba[:, 1].tolist()
    f3 = open('Predicted TMP/test_leave_y_' + gapped + leave_cell.lower() + '.pkl', 'rb')
    test_y = pkl.load(f3)
    test_y = np.array(test_y).tolist()

    total_total_predicted += predicted.tolist()
    total_total_predicted_proba += predicted_proba
    total_total_y += test_y

    print(leave_cell)
    roc = roc_auc_score(test_y, predicted_proba)
    print("ROC_AUC: " + str(roc))
    print('\n\n')

print('AVG')
roc = roc_auc_score(total_total_y, total_total_predicted_proba)
print("ROC_AUC: " + str(roc))
print('\n\n')
