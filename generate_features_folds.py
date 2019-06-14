import pandas as pd
import itertools
import re
import numpy as np


def gap_features(df, nucleotides):
    ret_def = pd.DataFrame()
    for i in range(1, len(df['sgRNA'][0]) - 2 + 1):
        for x1 in nucleotides:
            for x2 in nucleotides:
                col_name = 'GAP_' + x1 + '_' + str(i) + '_' + x2
                print('Generating Feature ' + col_name)
                ret_def[col_name] = pd.Series(data=(df.shape[0] * [0])).astype(np.int8)
                idx = 0
                for sgRNA in df['sgRNA']:
                    cnt = 0
                    for j in range(len(df['sgRNA'][0]) - (i + 1)):
                        if sgRNA[j] == x1 and sgRNA[j + i + 1] == x2:
                            cnt += 1
                    ret_def[col_name].at[idx] = np.int8(cnt)
                    idx += 1
    return ret_def


def position_independent(df, order, nucleotides):
    ret_def = pd.DataFrame()
    for ord_ in range(1, order + 1):
        for p in itertools.product(nucleotides, repeat=ord_):
            p = ''.join(p)
            ret_def[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int8)
            print('Generating Feature ' + p)
            idx = 0
            for sgRNA in df['sgRNA']:
                cnt = sgRNA.count(p)
                ret_def[p].at[idx] = np.int8(cnt)
                idx += 1
    return ret_def


def position_specific(df, order, nucleotides):
    ret_def = pd.DataFrame()
    for ord_ in range(1, order + 1):
        for p in itertools.product(nucleotides, repeat=ord_):
            p = ''.join(p)
            for i in range(0, len(df['sgRNA'][0]) - ord_ + 1):
                col_name = p + '_' + str(i + 1)
                ret_def[col_name] = pd.Series(data=(df.shape[0] * [0])).astype(np.int8)
            print('Finding positions for ' + p)
            idx = 0
            for sgRNA in df['sgRNA']:
                for m in re.finditer('(?=' + p + ')', sgRNA):
                    col_name = p + '_' + str(m.start() + 1)
                    ret_def[col_name].at[idx] = np.int8(1)
                idx += 1
    return ret_def


nucleotides_ = ['A', 'C', 'T', 'G']
cells = ['hct116', 'hek293', 'hela', 'hl60']

for gapped in [True, False]:
    for k in range(1, 6):
        for cell in cells:
            df_ = pd.read_csv('Folds/train_' + cell + '_' + str(k) + '.csv', delimiter=',')
            labels = pd.DataFrame(df_['label'].astype(np.int8), columns=['label'])
            df_pos_ind = position_independent(df_, 4, nucleotides_).astype(np.int8)
            df_pos_spe = position_specific(df_, 4, nucleotides_).astype(np.int8)
            if gapped:
                df_gap = gap_features(df_, nucleotides_).astype(np.int8)
                df_main = pd.concat([df_pos_ind, df_pos_spe, df_gap], axis=1, sort=False).astype(np.int8)
                df_main.to_hdf('Folds/train_x_' + cell + '_' + str(k) + '.h5', key=cell)
                labels.to_hdf('Folds/train_y_' + cell + '_' + str(k) + '.h5', key=cell)
            else:
                df_main = pd.concat([df_pos_ind, df_pos_spe], axis=1, sort=False).astype(np.int8)
                df_main.to_hdf('Folds/train_x_without_gapped_' + cell + '_' + str(k) + '.h5', key=cell)
                labels.to_hdf('Folds/train_y_without_gapped_' + cell + '_' + str(k) + '.h5', key=cell)

            df_ = pd.read_csv('Folds/test_' + cell + '_' + str(k) + '.csv', delimiter=',')
            labels = pd.DataFrame(df_['label'].astype(np.int8), columns=['label'])
            df_pos_ind = position_independent(df_, 4, nucleotides_).astype(np.int8)
            df_pos_spe = position_specific(df_, 4, nucleotides_).astype(np.int8)
            if gapped:
                df_gap = gap_features(df_, nucleotides_).astype(np.int8)
                df_main = pd.concat([df_pos_ind, df_pos_spe, df_gap], axis=1, sort=False).astype(np.int8)
                df_main.to_hdf('Folds/test_x_' + cell + '_' + str(k) + '.h5', key=cell)
                labels.to_hdf('Folds/test_y_' + cell + '_' + str(k) + '.h5', key=cell)
            else:
                df_main = pd.concat([df_pos_ind, df_pos_spe], axis=1, sort=False).astype(np.int8)
                df_main.to_hdf('Folds/test_x_without_gapped_' + cell + '_' + str(k) + '.h5', key=cell)
                labels.to_hdf('Folds/test_y_without_gapped_' + cell + '_' + str(k) + '.h5', key=cell)
