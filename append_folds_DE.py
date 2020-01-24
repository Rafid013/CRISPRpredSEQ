import pandas as pd
import numpy as np

cells = ['hct116', 'hek293', 'hela', 'hl60']

for gapped in ["without_gapped_", ""]:
    for i in range(1, 6):
        train_x = pd.DataFrame()
        train_y = pd.DataFrame()
        for leave_out_cell in cells:
            for cell in cells:
                if cell != leave_out_cell:
                    key = cell
                    train_x.append(pd.DataFrame(pd.read_hdf('Folds/train_x_' + gapped + cell.lower() +
                                                            '_' + str(i) + '.h5', key=key)),
                                   ignore_index=True).astype(np.int8)
                    train_y.append(pd.DataFrame(pd.read_hdf('Folds/train_y_' + gapped + cell.lower() +
                                                            '_' + str(i) + '.h5', key=key)).iloc[:, 0],
                                   ignore_index=True).astype(np.int8)

            train_x.to_hdf('Leave Folds/train_x_' + gapped + '_leave_' + leave_out_cell.lower() + '_' + str(i) + '.h5',
                           key='leave')
            train_y.to_hdf('Leave Folds/train_y_' + gapped + '_leave_' + leave_out_cell.lower() + '_' + str(i) + '.h5',
                           key='leave')
