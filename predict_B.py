import pickle as pkl
import pandas as pd
import sys
import os
import numpy as np


filepath = sys.argv[1]
filename = os.path.splitext(os.path.basename(filepath))[0]

pi = pd.DataFrame(pd.read_hdf('Data/' + filename + '_pi.h5', key='pi')).astype(np.int8)
ps = pd.DataFrame(pd.read_hdf('Data/' + filename + '_ps.h5', key='ps')).astype(np.int8)
# labels = pd.DataFrame(pd.read_hdf('Data/' + filename + '_labels.h5', key='labels')).astype(np.int8)

features = pd.concat([pi, ps], axis=1, sort=False).astype(np.int8)

f = open('Saved Models/' + filename + '_B.pkl', 'rb')
pipeline = pkl.load(f)

predictions = pd.DataFrame(pipeline.predict(features), columns=['prediction'])
predictions.to_csv('Data/' + filename + '_B_prediction.csv', sep=',', index=False)
