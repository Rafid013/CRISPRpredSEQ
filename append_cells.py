import pandas as pd
import numpy as np


features_hct116 = pd.DataFrame(pd.read_hdf('Data/features_hct116.h5', key='hct116')).astype(np.int8)
features_hek293 = pd.DataFrame(pd.read_hdf('Data/features_hek293.h5', key='hek293')).astype(np.int8)
features_hela = pd.DataFrame(pd.read_hdf('Data/features_hela.h5', key='hela')).astype(np.int8)
features_hl60 = pd.DataFrame(pd.read_hdf('Data/features_hl60.h5', key='hl60')).astype(np.int8)

labels_hct116 = pd.DataFrame(pd.read_hdf('Data/labels_hct116.h5', key='hct116')).astype(np.int8)
labels_hek293 = pd.DataFrame(pd.read_hdf('Data/labels_hek293.h5', key='hek293')).astype(np.int8)
labels_hela = pd.DataFrame(pd.read_hdf('Data/labels_hela.h5', key='hela')).astype(np.int8)
labels_hl60 = pd.DataFrame(pd.read_hdf('Data/labels_hl60.h5', key='hl60')).astype(np.int8)

features = features_hct116.append(features_hek293, ignore_index=True)
del features_hct116
del features_hek293
features = features.append(features_hela, ignore_index=True)
del features_hela
features = features.append(features_hl60, ignore_index=True)

labels = labels_hct116.append(labels_hek293, ignore_index=True)
del labels_hct116
del labels_hek293
labels = labels.append(labels_hela, ignore_index=True)
del labels_hela
labels = labels.append(labels_hl60, ignore_index=True)

print(features.shape)
features.to_hdf('Data/features_all.h5', key='all')
labels.to_hdf('Data/labels_all.h5', key='all')
