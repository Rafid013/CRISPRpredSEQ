import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import pickle as pkl


experiment = 'cv_without_gapped'  # 'cv', 'cv_without_gapped', 'crisprpred'
gapped = 'without_gapped_'  # 'without_gapped_', ''
for i in range(1, 6):
    test_x_list = [pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + 'all_' + str(i) + '.h5')),
                   pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + 'hct116_' + str(i) + '.h5', key='hct116')),
                   pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + 'hek293_' + str(i) + '.h5', key='hek293')),
                   pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + 'hela_' + str(i) + '.h5', key='hela')),
                   pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + 'hl60_' + str(i) + '.h5', key='hl60'))]

    test_y_list = [pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + 'all_' + str(i) + '.h5')).iloc[:, 0],
                   pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + 'hct116_' + str(i) + '.h5',
                                            key='hct116')).iloc[:, 0],
                   pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + 'hek293_' + str(i) + '.h5',
                                            key='hek293')).iloc[:, 0],
                   pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + 'hela_' + str(i) + '.h5',
                                            key='hela')).iloc[:, 0],
                   pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + 'hl60_' + str(i) + '.h5',
                                            key='hl60')).iloc[:, 0]]

    cells = ['ALL', 'HCT116', 'HEK293', 'HELA', 'HL60']

    f = open('Saved Models/trained_' + experiment + '_' + str(i) + '.pkl', 'rb')
    model = pkl.load(f)

    f_log = open('Logs/test_' + experiment + '_log_' + str(i) + '.txt', 'w')

    for test_x, test_y, cell in zip(test_x_list, test_y_list, cells):

        predicted = model.predict(test_x)
        acc = accuracy_score(test_y, predicted)
        pre = precision_score(test_y, predicted)
        rec = recall_score(test_y, predicted)
        predicted_proba = model.predict_proba(test_x)
        roc = roc_auc_score(test_y, predicted_proba[:, 1])
        f_log.write(cell + '\n')
        f_log.write("TEST ROC_AUC: " + str(roc) + "\n")
        f_log.write("TEST ACCURACY: " + str(acc) + "\n")
        f_log.write("TEST PRECISION: " + str(pre) + "\n")
        f_log.write("TEST RECALL: " + str(rec) + "\n")
        f_log.write('.........\n\n\n\n\n\n\n')

        f_predicted = open('Predicted/' + experiment + '_predicted_' + cell.lower() + '_' + str(i) + '.pkl', 'wb')
        pkl.dump(predicted, f_predicted)

        f_predicted_proba = open('Predicted/' + experiment + '_predicted_proba_' + cell.lower() + '_' + str(i) + '.pkl',
                                 'wb')
        pkl.dump(predicted_proba, f_predicted_proba)
