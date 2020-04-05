import pandas as pd


cells = ['hct116', 'hek293', 'hela', 'hl60']
pis = []
pss = []
gaps = []
labels = []

for cell in cells:
    pis.append(pd.DataFrame(pd.read_hdf('Data/' + cell + '_pi.h5', key='pi')))
    pss.append(pd.DataFrame(pd.read_hdf('Data/' + cell + '_ps.h5', key='ps')))
    gaps.append(pd.DataFrame(pd.read_hdf('Data/' + cell + '_gap.h5', key='gap')))
    labels.append(pd.DataFrame(pd.read_hdf('Data/' + cell + '_labels.h5', key='labels')))

pi = pis[0]
ps = pss[0]
gap = gaps[0]
label = labels[0]

for i in range(1, 4):
    pi = pi.append(pis[i], ignore_index=True)
    ps = ps.append(pss[i], ignore_index=True)
    gap = gap.append(gaps[i], ignore_index=True)
    label = label.append(labels[i], ignore_index=True)

print(pi)
print(ps)
print(gap)
print(label)

pi.to_hdf('Data/data_pi.h5', key='pi')
ps.to_hdf('Data/data_ps.h5', key='ps')
gap.to_hdf('Data/data_gap.h5', key='gap')
label.to_hdf('Data/data_labels.h5', key='labels')
