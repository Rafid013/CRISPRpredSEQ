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

cells = ['HCT116', 'HEK293', 'HELA', 'HL60']

total_total_predicted = []
total_total_predicted_proba = []
total_total_y = []
for leave_cell in cells:
    total_predicted = []
    total_predicted_proba = []
    total_y = []
    for i in range(1, 6):
        f1 = open('Predicted/' + experiment + '_predicted_' + leave_cell.lower() + '_' + str(i) + '.pkl', 'rb')
        f2 = open('Predicted/' + experiment + '_predicted_proba_' + leave_cell.lower() + '_' + str(i) + '.pkl', 'rb')
        predicted = pkl.load(f1)
        predicted_proba = pkl.load(f2)
        total_predicted += predicted.tolist()
        total_predicted_proba += predicted_proba[:, 1].tolist()
        f3 = open('Predicted/test_leave_y_' + gapped + leave_cell.lower() + '_' + str(i) + '.pkl', 'rb')
        test_y = pkl.load(f3)
        total_y += np.array(test_y).tolist()

    total_total_predicted += total_predicted
    total_total_predicted_proba += total_predicted_proba
    total_total_y += total_y

    print(leave_cell)
    roc = roc_auc_score(total_y, total_predicted_proba)
    print("ROC_AUC: " + str(roc))
    print('\n\n')

print('AVG')
roc = roc_auc_score(total_total_y, total_total_predicted_proba)
print("ROC_AUC: " + str(roc))
print('\n\n')
