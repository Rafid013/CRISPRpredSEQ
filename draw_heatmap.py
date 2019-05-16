import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


df1 = pd.read_csv('Data/hct116.csv', delimiter=',')
df2 = pd.read_csv('Data/hek293.csv', delimiter=',')
df3 = pd.read_csv('Data/hela.csv', delimiter=',')
df4 = pd.read_csv('Data/hl60.csv', delimiter=',')

ret_df = pd.DataFrame(columns=['Cells', 'Nucleotides', 'COUNT'])

count_hct116_A = count_hct116_C = count_hct116_T = count_hct116_G = 0.0
for val in df1.values:
    sgRNA = val[0]
    count_hct116_A += sgRNA.count('A')
    count_hct116_C += sgRNA.count('C')
    count_hct116_T += sgRNA.count('T')
    count_hct116_G += sgRNA.count('G')

total_hct116_nt = count_hct116_A + count_hct116_C + count_hct116_T + count_hct116_G
count_hct116_A /= total_hct116_nt
count_hct116_C /= total_hct116_nt
count_hct116_T /= total_hct116_nt
count_hct116_G /= total_hct116_nt

ret_df.loc[0] = ['HCT116', 'A', count_hct116_A]
ret_df.loc[1] = ['HCT116', 'C', count_hct116_C]
ret_df.loc[2] = ['HCT116', 'T', count_hct116_T]
ret_df.loc[3] = ['HCT116', 'G', count_hct116_G]

count_hek293_A = count_hek293_C = count_hek293_T = count_hek293_G = 0.0
for val in df2.values:
    sgRNA = val[0]
    count_hek293_A += sgRNA.count('A')
    count_hek293_C += sgRNA.count('C')
    count_hek293_T += sgRNA.count('T')
    count_hek293_G += sgRNA.count('G')

total_hek293_nt = count_hek293_A + count_hek293_C + count_hek293_T + count_hek293_G
count_hek293_A /= total_hek293_nt
count_hek293_C /= total_hek293_nt
count_hek293_T /= total_hek293_nt
count_hek293_G /= total_hek293_nt

ret_df.loc[4] = ['HEK293', 'A', count_hek293_A]
ret_df.loc[5] = ['HEK293', 'C', count_hek293_C]
ret_df.loc[6] = ['HEK293', 'T', count_hek293_T]
ret_df.loc[7] = ['HEK293', 'G', count_hek293_G]

count_hela_A = count_hela_C = count_hela_T = count_hela_G = 0.0
for val in df3.values:
    sgRNA = val[0]
    count_hela_A += sgRNA.count('A')
    count_hela_C += sgRNA.count('C')
    count_hela_T += sgRNA.count('T')
    count_hela_G += sgRNA.count('G')

total_hela_nt = count_hela_A + count_hela_C + count_hela_T + count_hela_G
count_hela_A /= total_hela_nt
count_hela_C /= total_hela_nt
count_hela_T /= total_hela_nt
count_hela_G /= total_hela_nt

ret_df.loc[8] = ['HELA', 'A', count_hela_A]
ret_df.loc[9] = ['HELA', 'C', count_hela_C]
ret_df.loc[10] = ['HELA', 'T', count_hela_T]
ret_df.loc[11] = ['HELA', 'G', count_hela_G]

count_hl60_A = count_hl60_C = count_hl60_T = count_hl60_G = 0.0
for val in df4.values:
    sgRNA = val[0]
    count_hl60_A += sgRNA.count('A')
    count_hl60_C += sgRNA.count('C')
    count_hl60_T += sgRNA.count('T')
    count_hl60_G += sgRNA.count('G')

total_hl60_nt = count_hl60_A + count_hl60_C + count_hl60_T + count_hl60_G
count_hl60_A /= total_hl60_nt
count_hl60_C /= total_hl60_nt
count_hl60_T /= total_hl60_nt
count_hl60_G /= total_hl60_nt

ret_df.loc[12] = ['HL60', 'A', count_hl60_A]
ret_df.loc[13] = ['HL60', 'C', count_hl60_C]
ret_df.loc[14] = ['HL60', 'T', count_hl60_T]
ret_df.loc[15] = ['HL60', 'G', count_hl60_G]

ret_df = ret_df.pivot(index='Cells', columns='Nucleotides', values='COUNT')
ax = sns.heatmap(ret_df, annot=False)
plt.savefig('Figures/heat_map.png')
