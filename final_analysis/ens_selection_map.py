import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates


from lib import config


VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']

# VALID_STATES = ["MD","AL","GA","LA","AK","MN","KS","TX","WI","PA","OH","VA","AZ","NC","TN",
#                 "KY","SC","NV","AR","ID","DE","UT","OR","SD","NM","WA","ND",
#                 "NE","VT","RI","WV","MI","MA","NH","ME","NJ","NY"]


ens_ids = []
for i, state in enumerate(VALID_STATES):
    print state

    state_dat = pd.read_csv(config.STATES_DIR + '/{0}/top_ens_preds.csv'.format(state), parse_dates=[0])

    argo = state_dat['ARGO(gt,ath)']
    net = state_dat['Net']
    ens = state_dat['ARGONet']

    times = state_dat.Week

    ids = np.zeros(state_dat.shape[0])
    for j in range(state_dat.shape[0]):
        if np.allclose(ens[j], argo[j]):
            ids[j] = 1
        elif np.allclose(ens[j], net[j]):
            ids[j] = 2
        else:
            print "mismatch"
    ens_ids.append(ids)

ens_ids = np.vstack((ens_ids))


# # initalize figure and axes
grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
fig.set_size_inches(15, 7)

# fig,ax=plt.subplots()
# fig.set_size_inches(14, 20)

# colormap
colmap = ListedColormap(['#8196d3', '#bfd850'])

ax = seaborn.heatmap(ens_ids, ax=ax, cmap=colmap, yticklabels=VALID_STATES, xticklabels=np.arange(2015,2018),
                     cbar_ax=cbar_ax, cbar_kws={'orientation':'vertical'}, square=False,
                     linecolor="white", linewidths=0.5)

# heatmap = ax.pcolor(ens_ids, cmap=colmap)
# plt.yticks(np.arange(0.5, ens_ids.shape[0], 1), VALID_STATES)
ax.set_xticks(np.arange(14, ens_ids.shape[1], 52))
# ax.set_xticks(np.arange(2014,2018)*ax.get_xlim()[1])

ax.set_title('Model selected for ARGONet over time', fontsize=12)
ax.tick_params(axis='y', length=0, labelsize=8)
ax.tick_params(axis='x', length=0, labelsize=8, rotation=90)
#
# # Customize tick marks and positions
cbar_ax.set_yticklabels(['ARGO', 'Net'])
cbar_ax.yaxis.set_ticks([ 0.25, 0.75])
cbar_ax.tick_params(axis='y', length=0, labelsize=8)


fig.tight_layout()
fig.savefig('ens_select_map.png', dpi=200)


# plt.show()
