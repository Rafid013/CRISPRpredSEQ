from thundersvm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
import pickle as pkl
from scipy import stats

cas9_type = int(input("Cas9 Type?\n"))
if cas9_type == 1:
    cas9 = 'WT-SpCas9'
elif cas9_type == 2:
    cas9 = 'eSpCas9'
else:
    cas9 = 'SpCas9-HF1'

features = pd.read_hdf('DEEPHF/data_without_gapped_x.h5', key='deephf')
labels = pd.read_hdf('DEEPHF/data_y' + str(cas9_type) + '.h5', key='deephf')

data = pd.concat([features, labels], axis=1, ignore_index=True)

train_data, test_data = train_test_split(data, test_size=0.15, random_state=1)

train_data = train_data.dropna().reset_index(drop=True)

test_data = test_data.dropna().reset_index(drop=True)

extraTree = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=1)

steps = [('SFM', SelectFromModel(estimator=extraTree)),
         ('scaler', StandardScaler()),
         ('SVM', SVR(C=10, gamma=0.001, kernel='rbf', cache_size=20000, verbose=True,
                     max_mem_size=6000))]

train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]
test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, -1]
model = Pipeline(steps)
model.fit(train_x, train_y)

predict = model.predict(test_x)

print(cas9)
print(stats.spearmanr(test_y, predict))

sfm = model['SFM']
trained_rf = sfm.estimator_
scaler = model['scaler']
svm = model['SVM']

f = open('DEEPHF Models/B_sfm_' + str(cas9_type) + '.pkl', 'wb')
pkl.dump(sfm, f)

f = open('DEEPHF Models/B_rf_' + str(cas9_type) + '.pkl', 'wb')
pkl.dump(trained_rf, f)

f = open('DEEPHF Models/B_scaler_' + str(cas9_type) + '.pkl', 'wb')
pkl.dump(scaler, f)

svm.save_to_file('DEEPHF Models/B_svm_' + str(cas9_type) + '.txt')
