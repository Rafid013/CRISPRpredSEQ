import pandas as pd
import re


result_roc = pd.DataFrame()
result_roc['C\\gamma'] = [1, 10, 100]
result_roc['0.0001'] = [0]*3
result_roc['0.001'] = [0]*3
result_roc['0.01'] = [0]*3


exp_type = input("Which experiment? B or C\n")
exp_type = exp_type.upper()
if exp_type == 'B':
    gapped = 'without_gapped_'
else:
    gapped = ''

for i in range(1, 6):
    f = open('Logs/cross_validation_log_' + gapped + str(i) + '.txt', 'r')
    text = f.read()
    # noinspection PyRedeclaration
    lines = text.split('\n')[6:]

    for j in range(9):
        line_block = lines[:9]
        lines = lines[12:]

        roc = float(re.findall("\d+\.\d+|\d+", line_block[0])[0])
        acc = float(re.findall("\d+\.\d+|\d+", line_block[2])[0])
        pre = float(re.findall("\d+\.\d+|\d+", line_block[4])[0])
        rec = float(re.findall("\d+\.\d+|\d+", line_block[6])[0])
        hyperparameters = re.findall("\d+\.\d+|\d+", line_block[8])
        C_param = float(hyperparameters[0])
        gamma_param = hyperparameters[1]
        result_roc.loc[result_roc['C\\gamma'] == C_param, gamma_param] += roc

for C_param in [1, 10, 100]:
    for gamma_param in [0.0001, 0.001, 0.01]:
        result_roc.loc[result_roc['C\\gamma'] == C_param, str(gamma_param)] /= 5

result_roc.to_csv('Results/gridsearch_roc_' + gapped + 'result.csv', sep=',', index=False)
