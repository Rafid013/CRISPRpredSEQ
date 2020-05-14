import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


df = pd.read_csv('Results/final_test_results.csv', delimiter=',')
columns = np.array(df.iloc[:, 0])
df = np.array(df.T.iloc[1:, :])
to_draw_df = pd.DataFrame(data=df, columns=columns)
to_draw_df['Cells'] = pd.Series(['HCT116', 'HEK293', 'HELA', 'HL60', 'ALL'])
ax = to_draw_df.plot.bar(x='Cells', figsize=(14, 13))
fontP = FontProperties()
fontP.set_size('medium')
ax.legend(loc='upper right', prop=fontP, ncol=2)
ax.set_ylabel('ROC-AUC')
ax.set_ylim(0, 1)

plt.savefig('Figures/roc_comparison_ac.eps')


df = pd.read_csv('Results/final_loc_results.csv', delimiter=',')
columns = np.array(df.iloc[:, 0])
df = np.array(df.T.iloc[1:, :])
to_draw_df = pd.DataFrame(data=df, columns=columns)
to_draw_df['Cells'] = pd.Series(['Leave HCT116', 'Leave HELA', 'Leave HL60'])
ax = to_draw_df.plot.bar(x='Cells', figsize=(14, 13))
fontP = FontProperties()
fontP.set_size('medium')
ax.legend(loc='upper right', prop=fontP, ncol=2)
ax.set_ylabel('ROC-AUC')
ax.set_ylim(0, 1)

plt.savefig('Figures/roc_comparison_df.eps')
