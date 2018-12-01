import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


from lib import config


def PEARSON(predictions, targets):
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]


data = pd.read_csv(config.DATA_DIR + '/analysis/geomap_data.csv')

data.head()


fig, ax = plt.subplots(nrows=2, ncols=3)
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[0,2]
ax4 = ax[1,0]
ax5 = ax[1,1]
ax6 = ax[1,2]


##################### 1 #########################
tmp = data[data.State!='Wisconsin']
tmp2 = data[data.State=='Wisconsin']
x = tmp['athena coverage'].values
y = tmp['improvement3'].values
fit = np.polyfit(x, y, deg=1)
Rval = PEARSON(x,y) ** 2

ax1.plot(x, fit[0] * x + fit[1], color='#e4595b')
ax1.scatter(tmp['athena coverage'], tmp['improvement3'], color='#4393c3')
ax1.scatter(tmp2['athena coverage'], tmp2['improvement3'], color='#e4595b')
ax1.set_ylabel("ARGONet improvement (%)", fontsize=10)
ax1.set_xlabel("athenahealth coverage", fontsize=10)
ax1.set_ylim([-1,48])
ax1.set_xlim([-1,19])
ax1.set_yticks([0,15,30,45])
ax1.set_xticks([0,6,12,18])
ax1.spines['bottom'].set_bounds(0, 18)
ax1.spines['left'].set_bounds(0, 45)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.annotate('$R^2= %.2f$' % Rval, xy=(12,9), fontsize=10)


##################### 2 #########################
x = data['GT terms'].values
y = data['improvement3'].values
fit = np.polyfit(x, y, deg=1)
Rval = PEARSON(x,y) ** 2

ax2.plot(x, fit[0] * x + fit[1], color='#e4595b')
ax2.scatter(x, y, color='#4393c3')
ax2.set_ylabel("ARGONet improvement (%)", fontsize=10)
ax2.set_xlabel("Non-zero GT terms", fontsize=10)
ax2.set_ylim([-1,48])
ax2.set_xlim([-1,190])
ax2.set_yticks([0,15,30,45])
ax2.set_xticks([0,60,120,180])
ax2.spines['bottom'].set_bounds(0, 180)
ax2.spines['left'].set_bounds(0, 45)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.annotate('$R^2= %.2f$' % Rval, xy=(130,9), fontsize=10)


##################### 3 #########################
x = data['Population'].values
y = data['GT terms'].values
fit = np.polyfit(x, y, deg=1)
Rval = PEARSON(x,y) ** 2

ax3.plot(x, fit[0] * x + fit[1], color='#e4595b')
ax3.scatter(x, y, color='#4393c3')
ax3.set_ylabel("Non-zero GT terms", fontsize=10)
ax3.set_xlabel("Population (millions)", fontsize=10)
ax3.set_xlim([-1,24])
ax3.set_ylim([-1,192])
ax3.set_xticks([0,8,16,24])
ax3.set_yticks([0,60,120,180])
ax3.spines['bottom'].set_bounds(0, 24)
ax3.spines['left'].set_bounds(0, 180)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.annotate('$R^2= %.2f$' % Rval, xy=(18,35), fontsize=10)


##################### 4 #########################
x = data['deviation'].values
y = data['improvement3'].values
fit = np.polyfit(x, y, deg=1)
Rval = PEARSON(x,y) ** 2

ax4.plot(x, fit[0] * x + fit[1], color='#e4595b')
ax4.scatter(x, y, color='#4393c3')
ax4.set_ylabel("ARGONet improvement (%)", fontsize=10)
ax4.set_xlabel("Deviation: $\sigma(Y)/E(Y)$", fontsize=10)
ax4.set_ylim([-1,48])
ax4.set_xlim([0.3, 1.65])
ax4.set_yticks([0,15,30,45])
ax4.set_xticks([0.4, 0.8, 1.2, 1.6])
ax4.spines['bottom'].set_bounds(0.4, 1.6)
ax4.spines['left'].set_bounds(0, 45)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.annotate('$R^2= %.2f$' % Rval, xy=(1.3,25), fontsize=10)


##################### 5 #########################
x = data['providers'].values
y = data['improvement3'].values
fit = np.polyfit(x, y, deg=1)
Rval = PEARSON(x,y) ** 2

ax5.plot(x, fit[0] * x + fit[1], color='#e4595b')
ax5.scatter(x, y, color='#4393c3')
ax5.set_ylabel("ARGONet improvement (%)", fontsize=10)
ax5.set_xlabel("Providers", fontsize=10)
ax5.set_ylim([-1,48])
ax5.set_xlim([-10,140])
ax5.set_yticks([0,15,30,45])
ax5.set_xticks([0,40,80,120])
ax5.spines['bottom'].set_bounds(0, 120)
ax5.spines['left'].set_bounds(0, 45)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.annotate('$R^2= %.2f$' % Rval, xy=(100,9), fontsize=10)


##################### 6 #########################
ax6.axis('off')


# plt.show()
fig.set_size_inches(15, 8)
fig.savefig('scatter.png', dpi=300)
