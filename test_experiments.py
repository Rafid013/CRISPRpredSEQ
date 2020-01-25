import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import pickle as pkl

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

for i in range(1, 6):
    test_x_list = [pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + 'all_' + str(i) + '.h5', key='all')),
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

    test_list = [pd.concat([test_x_list[0], test_y_list[0]], axis=1),
                 pd.concat([test_x_list[1], test_y_list[1]], axis=1),
                 pd.concat([test_x_list[2], test_y_list[2]], axis=1),
                 pd.concat([test_x_list[3], test_y_list[3]], axis=1),
                 pd.concat([test_x_list[4], test_y_list[4]], axis=1)]

    train_x_list = [pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'all_' + str(i) + '.h5')),
                    pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'hct116_' + str(i) + '.h5', key='hct116')),
                    pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'hek293_' + str(i) + '.h5', key='hek293')),
                    pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'hela_' + str(i) + '.h5', key='hela')),
                    pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + 'hl60_' + str(i) + '.h5', key='hl60'))]

    train_y_list = [pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'all_' + str(i) + '.h5')).iloc[:, 0],
                    pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'hct116_' + str(i) + '.h5',
                                             key='hct116')).iloc[:, 0],
                    pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'hek293_' + str(i) + '.h5',
                                             key='hek293')).iloc[:, 0],
                    pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'hela_' + str(i) + '.h5',
                                             key='hela')).iloc[:, 0],
                    pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + 'hl60_' + str(i) + '.h5',
                                             key='hl60')).iloc[:, 0]]

    train_list = [pd.concat([train_x_list[0], train_y_list[0]], axis=1),
                  pd.concat([train_x_list[1], train_y_list[1]], axis=1),
                  pd.concat([train_x_list[2], train_y_list[2]], axis=1),
                  pd.concat([train_x_list[3], train_y_list[3]], axis=1),
                  pd.concat([train_x_list[4], train_y_list[4]], axis=1)]

    test_x_list = []
    test_y_list = []
    for train, test in zip(train_list, test_list):
        train_set = set([tuple(line) for line in train.values])
        test_set = set([tuple(line) for line in test.values])
        test_set = test_set.difference(train_set)
        test_tmp = pd.DataFrame(list(test_set))
        test_x_list.append(test_tmp.iloc[:, :-1])
        test_y_list.append(test_tmp.iloc[:, -1])

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

        f_test_y = open('Predicted/test_y_' + gapped + cell.lower() + '_' + str(i) + '.pkl', 'wb')
        pkl.dump(test_y, f_test_y)
