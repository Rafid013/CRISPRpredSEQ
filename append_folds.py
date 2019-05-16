import pandas as pd
import numpy as np


gapped = "without_gapped_"  # "without_gapped_" or ""
for i in range(1, 2):
    cell = 'hct116'
    key = cell
    train_x1 = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key))
    train_y1 = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key)).iloc[:, 0]

    test_x1 = pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key))
    test_y1 = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key)).iloc[:, 0]

    cell = 'hek293'
    key = cell
    train_x2 = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key))
    train_y2 = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key)).iloc[:, 0]

    test_x2 = pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key))
    test_y2 = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key)).iloc[:, 0]

    cell = 'hela'
    key = cell
    train_x3 = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key))
    train_y3 = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key)).iloc[:, 0]

    test_x3 = pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key))
    test_y3 = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key)).iloc[:, 0]

    cell = 'hl60'
    key = cell
    train_x4 = pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key))
    train_y4 = pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                        key=key)).iloc[:, 0]

    test_x4 = pd.DataFrame(pd.read_hdf('Folds/test_x_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key))
    test_y4 = pd.DataFrame(pd.read_hdf('Folds/test_y_' + gapped + cell.lower() + '_' + str(i) + '.h5',
                                       key=key)).iloc[:, 0]

    train_x = train_x1.append(train_x2, ignore_index=True)
    del train_x1
    del train_x2
    train_x = train_x.append(train_x3, ignore_index=True)
    del train_x3
    train_x = train_x.append(train_x4, ignore_index=True).astype(np.int8)
    del train_x4

    train_y = train_y1.append(train_y2, ignore_index=True)
    del train_y1
    del train_y2
    train_y = train_y.append(train_y3, ignore_index=True)
    del train_y3
    train_y = train_y.append(train_y4, ignore_index=True)
    del train_y4

    test_x = test_x1.append(test_x2, ignore_index=True)
    del test_x1
    del test_x2
    test_x = test_x.append(test_x3, ignore_index=True)
    del test_x3
    test_x = test_x.append(test_x4, ignore_index=True)
    del test_x4

    test_y = test_y1.append(test_y2, ignore_index=True)
    del test_y1
    del test_y2
    test_y = test_y.append(test_y3, ignore_index=True)
    del test_y3
    test_y = test_y.append(test_y4, ignore_index=True)
    del test_y4

    cell = 'all'

    train_x.to_hdf('Folds/train_x_' + gapped + cell.lower() + '_' + str(i) + '.h5', key='all')
    train_y.to_hdf('Folds/train_y_' + gapped + cell.lower() + '_' + str(i) + '.h5', key='all')

    test_x.to_hdf('Folds/test_x_' + gapped + cell.lower() + '_' + str(i) + '.h5', key='all')
    test_y.to_hdf('Folds/test_y_' + gapped + cell.lower() + '_' + str(i) + '.h5', key='all')
