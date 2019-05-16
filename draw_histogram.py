import pandas as pd
import matplotlib.pylab as plt


cells = ['hct116', 'hek293', 'hela', 'hl60']

for cell in cells:
    df = pd.read_csv('Data/' + cell + '.csv', delimiter=',')

    count_A = [0.0]*23
    count_C = [0.0]*23
    count_T = [0.0]*23
    count_G = [0.0]*23

    legend = ['A', 'C', 'T', 'G']

    for val in df.values:
        sgRNA = val[0]
        for i in range(23):
            nt = sgRNA[i]
            if nt == 'A':
                count_A[i] += 1
            elif nt == 'C':
                count_C[i] += 1
            elif nt == 'T':
                count_T[i] += 1
            else:
                count_G[i] += 1

    for i in range(23):
        total = count_A[i] + count_C[i] + count_T[i] + count_G[i]
        count_A[i] /= total
        count_C[i] /= total
        count_T[i] /= total
        count_G[i] /= total

    df = pd.DataFrame()
    df['Position'] = pd.Series(range(1, 24))
    df['A'] = pd.Series(count_A)
    df['C'] = pd.Series(count_C)
    df['T'] = pd.Series(count_T)
    df['G'] = pd.Series(count_G)
    ax = df.plot.bar(x='Position', width=0.75, title=cell.upper() + ' Counts')
    ax.set_ylabel('Normalized Count')
    plt.savefig('Figures/bar_chart_' + cell + '.png')
