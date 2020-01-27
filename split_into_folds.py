import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

cells = ['hct116', 'hek293', 'hela', 'hl60']

train_list = {'hct116': [], 'hek293': [], 'hela': [], 'hl60': [], 'all': []}
test_list = {'hct116': [], 'hek293': [], 'hela': [], 'hl60': []}
train_loc_list = {'hct116': [], 'hek293': [], 'hela': [], 'hl60': []}

# get main folds
for cell in cells:
    data_set = pd.read_csv('Data/' + cell + '.csv', delimiter=',')
    data_set_y = data_set['label']
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    i = 1
    for train_index, test_index in skf.split(data_set, data_set_y):
        train = pd.DataFrame(np.array(data_set)[train_index], columns=['sgRNA', 'label'])
        test = pd.DataFrame(np.array(data_set)[test_index], columns=['sgRNA', 'label'])

        train_list[cell].append(train)
        test_list[cell].append(test)

        train.to_csv('Folds/train_' + cell + '_' + str(i) + '.csv', index=False, sep=',')
        test.to_csv('Folds/test_' + cell + '_' + str(i) + '.csv', index=False, sep=',')
        i += 1

# append train data for all cells for each fold
for i in range(1, 6):
    train_all = pd.DataFrame(columns=['sgRNA', 'label'])
    for cell in cells:
        train_all = train_all.append(train_list[cell][i - 1], ignore_index=True)
    train_list['all'].append(train_all)
    train_all.to_csv('Folds/train_all_' + str(i) + '.csv', index=False, sep=',')


# append train data for all cells except one for each fold (leave one out)
for i in range(1, 6):
    for leave_cell in cells:
        train_loc = pd.DataFrame(columns=['sgRNA', 'label'])
        for cell in cells:
            if leave_cell != cell:
                train_loc = train_loc.append(train_list[cell][i - 1], ignore_index=True)
        train_loc_list[leave_cell].append(train_loc)
        train_loc.to_csv('Leave Folds/train_leave_' + leave_cell + '_' + str(i) + '.csv', index=False, sep=',')


# remove common data between train (leave one cell) and test (for left out cell) for each fold
for i in range(1, 6):
    for leave_cell in cells:
        test_loc = test_list[leave_cell][i - 1]
        train_loc = train_list[leave_cell][i - 1]
        train_set = set([tuple(line) for line in train_loc.values])
        test_set = set([tuple(line) for line in test_loc.values])
        test_set = test_set.difference(train_set)
        test = pd.DataFrame(list(test_set), columns=['sgRNA', 'label'])
        test.to_csv('Leave Folds/test_leave_' + leave_cell + '_' + str(i) + '.csv', index=False, sep=',')
