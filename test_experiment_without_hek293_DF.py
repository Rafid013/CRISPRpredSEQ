import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import pickle as pkl

exp_type = input("Which experiment?\n")
exp_type = exp_type.upper()
if exp_type == 'D' or exp_type == 'd':
    experiment = 'D'
    gapped = 'without_gapped_'
elif exp_type == 'E' or exp_type == 'e':
    experiment = 'E'
    gapped = 'without_gapped_'
else:
    experiment = 'F'
    gapped = ''

cells = ['HCT116', 'HELA', 'HL60']

for leave_cell in cells:
    f_log = open('TMP/test_' + experiment + '__leave_log_' + leave_cell.lower() + '.txt', 'w')

    test_x = pd.DataFrame(pd.read_hdf('TMP/test_leave_x_' + gapped +
                                      leave_cell.lower() + '.h5',
                                      key=leave_cell.lower()))

    test_y = pd.DataFrame(pd.read_hdf('TMP/test_leave_y_' + gapped +
                                      leave_cell.lower() + '.h5',
                                      key=leave_cell.lower()))

    f = open('TMP/trained_' + experiment + '_' + leave_cell.lower() + '.pkl', 'rb')
    model = pkl.load(f)

    predicted = model.predict(test_x)
    acc = accuracy_score(test_y, predicted)
    pre = precision_score(test_y, predicted)
    rec = recall_score(test_y, predicted)
    predicted_proba = model.predict_proba(test_x)
    roc = roc_auc_score(test_y, predicted_proba[:, 1])
    f_log.write("TEST ROC_AUC: " + str(roc) + "\n")
    f_log.write("TEST ACCURACY: " + str(acc) + "\n")
    f_log.write("TEST PRECISION: " + str(pre) + "\n")
    f_log.write("TEST RECALL: " + str(rec) + "\n")
    f_log.write('.........\n\n\n\n\n\n\n')

    f_predicted = open('Predicted TMP/' + experiment + '_predicted_' + leave_cell.lower() + '.pkl', 'wb')
    pkl.dump(predicted, f_predicted)

    f_predicted_proba = open('Predicted TMP/' + experiment + '_predicted_proba_' + leave_cell.lower() + '.pkl', 'wb')
    pkl.dump(predicted_proba, f_predicted_proba)

    f_test_y = open('Predicted TMP/test_leave_y_' + gapped + leave_cell.lower() + '.pkl', 'wb')
    pkl.dump(test_y, f_test_y)
