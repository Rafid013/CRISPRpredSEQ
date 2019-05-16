import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


cells = ['hct116', 'hek293', 'hela', 'hl60']

for cell in cells:
    data_set = pd.read_csv('Data/' + cell + '.csv', delimiter=',')
    data_set_y = data_set['label']
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    i = 1
    for train_index, test_index in skf.split(data_set, data_set_y):
        train = pd.DataFrame(np.array(data_set)[train_index], columns=['sgRNA', 'label'])
        test = pd.DataFrame(np.array(data_set)[test_index], columns=['sgRNA', 'label'])

        train.to_csv('Folds/train_' + cell + '_' + str(i) + '.csv', index=False, sep=',')
        test.to_csv('Folds/test_' + cell + '_' + str(i) + '.csv', index=False, sep=',')

        i += 1
