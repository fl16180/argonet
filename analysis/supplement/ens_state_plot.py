import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import matplotlib
import matplotlib.gridspec as gridspec

from lib import config

plt.style.use('seaborn-poster')


VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']



n_x = 3
n_y = 5

STATES_PER_PAGE = n_x * n_y

def get_selection_data():
    ens_ids = []
    for i, state in enumerate(VALID_STATES):
        # print state

        state_dat = pd.read_csv(config.STATES_DIR + '/{0}/argo_net_ens_preds.csv'.format(state), parse_dates=[0])

        argo = state_dat['ARGO']
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
    return ens_ids


def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]


def main_plot(STATES, page, ids):

    fig = plt.figure()

    # master grid (5 x 3 grid)
    widths = [3, 3, 3]
    heights = [8, 8, 8, 8, 8]
    spec = gridspec.GridSpec(ncols=n_x, nrows=n_y, width_ratios=widths,
                            height_ratios=heights, wspace=0.15, hspace=0.20)

    for i, state in enumerate(STATES):

        # for each gridspace define subgrid (2x1)
        row = int(i / n_x)
        col = i % n_x
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec[row, col], hspace=.1,
                                               height_ratios=[10,1])

        # create time series subplot
        ax = fig.add_subplot(gs2[0])

        models = pd.read_csv(config.STATES_DIR + '/{0}/argo_net_ens_preds.csv'.format(state), parse_dates=[0])

        models = models.set_index('Week')
        weeks = models.index.values
        target = models['ILI']
        ar = models['AR52']
        argo = models['ARGO']
        ens = models['ARGONet']

        ar[target == 0] = np.nan
        argo[target == 0] = np.nan
        ens[target == 0] = np.nan
        target[target == 0] = np.nan

        ar = smooth(ar, window_len=5, window='hanning')
        target = smooth(target, window_len=5, window='hanning')
        argo = smooth(argo, window_len=5, window='hanning')
        ens = smooth(ens, window_len=5, window='hanning')

        a, = ax.plot_date(x=weeks, y=argo, fmt="-", color='#8196d3', label='ARGO', linewidth=3, alpha=0.9)
        d, = ax.plot_date(x=weeks, y=ar, fmt="--", color='k', label='AR52', linewidth=2, alpha=0.35, dashes=(3,3))
        b, = ax.plot_date(x=weeks, y=target, fmt="k-", label=r'%ILI', linewidth=2.5, alpha=0.7)
        c, = ax.plot_date(x=weeks, y=ens, fmt="-", color='#e25658', label='ARGONet', linewidth=2.5)


        argo_patch = mpatches.Patch(color='#8196d3', label='ARGO selected')
        net_patch = mpatches.Patch(color='#bfd850', label='Net selected')

        if state in ['MI','SD']:
            l1 = ax.legend(handles=[b,d,a,c], loc='upper center',
                      frameon=False, bbox_to_anchor=(0.5, -0.5), ncol=4)
            l2 = ax.legend(handles=[argo_patch, net_patch], loc='upper center',
                      frameon=False, bbox_to_anchor=(0.5, -0.7), ncol=2)
            plt.gca().add_artist(l1)

        if state in ['WA']:
            l1 = ax.legend(handles=[b,d,a,c], loc='upper center',
                      frameon=False, bbox_to_anchor=(1, -0.6), ncol=4)
            l2 = ax.legend(handles=[argo_patch, net_patch], loc='upper center',
                      frameon=False, bbox_to_anchor=(1, -0.8), ncol=2)
            plt.gca().add_artist(l1)


        # create heatmap subplot
        ax2 = fig.add_subplot(gs2[1], sharex=ax)

        id_row = ids[i, :]
        id_row[np.isnan(argo)] = np.nan

        pad = np.zeros(len(id_row))
        id_row = np.vstack((pad, id_row))
        id_row[0, :] = np.nan

        colmap = ListedColormap(['#8196d3', '#bfd850'])

        # y_argo = (ens_ids[i, :] == 1).astype(float)
        # y_net = (ens_ids[i, :] == 2).astype(float)
        # y_argo[y_argo == 0] = np.nan
        # y_net[y_net == 0] = np.nan
        # dates = mdates.date2num(weeks)
        ax2.pcolor(weeks, np.arange(3), id_row, cmap=colmap)

        # ax2.spines['top'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        # ax2.spines['left'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # ax2.tick_params(axis='y', rotation=90, length=8, width=2, labelsize=13)


        # a, = ax2.plot_date(x=weeks, y=y_argo, fmt="-", color='#8196d3', label='ARGO', linewidth=3, alpha=0.75)
        # b, = ax2.plot_date(x=weeks, y=y_net, fmt="-", color='#bfd850', label='ARGO', linewidth=3, alpha=0.75)
        # ax2.set_xlim((735456.05, 736510.95))



        #### manage shared axes ####
        # ax.xaxis.set_major_locator(mdates.YearLocator())

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(2)
        # ax.Axes.tick_params()

        if state in ['AK','AL','AZ']:
            ax.text(.47,0.86,state,
                horizontalalignment='center',
                fontsize=16,
                transform=ax.transAxes)
        else:
            ax.text(.50,0.86,state,
                horizontalalignment='center',
                fontsize=16,
                transform=ax.transAxes)

        ax.tick_params(axis='y', rotation=90, length=8, width=2, labelsize=12)
        ax.locator_params(axis='y', tight=True, nbins=4)

        ax.set_xticks([])
        ax.tick_params(axis='x', length=0, width=0)
        ax2.set_yticks([])

        if i < n_x * (n_y - 1):
            ax.axes.xaxis.set_ticklabels([])
        if np.mod(i, n_x) == 0:
            ax.set_ylabel('%ILI', fontsize=13)

        txs = ax.yaxis.get_majorticklocs()
        if state in ['LA', 'AR']:
            ax.spines['left'].set_bounds(0, 7.5)
        else:
            ax.spines['left'].set_bounds(0, txs[-2])
        txs2 = ax.xaxis.get_majorticklocs()
        # print txs2
        # print ax.get_xlim()

        # hspace = np.abs(txs[1] - txs[0]) / 10
        ax.set_ylim(bottom=0)

        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.tick_params(axis='x', length=8, width=2, labelsize=12)
        ax2.spines['bottom'].set_bounds(735599, 736330)
        ax2.set_xlim((735456.05, 736510.95))
        ax2.spines['bottom'].set_linewidth(2)


        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)



    # plt.legend(loc='best', handles=[a,b,c,d])

    # plt.subplots_adjust(wspace=0.15, hspace=0.25)


    fig = plt.gcf()
    fig.set_size_inches(16, 20)
    fig.savefig(config.HOME_DIR + '/analysis/ens_states_{0}.png'.format(page), format='png', dpi=300)

    #plt.show()




if __name__ == '__main__':

    ens_ids = get_selection_data()

    batch_intervals = range(0, len(VALID_STATES), STATES_PER_PAGE)

    for i, batch_start in enumerate(batch_intervals):
        batch_end = min(batch_start + STATES_PER_PAGE, len(VALID_STATES))
        batch = VALID_STATES[batch_start:batch_end]

        print i
        main_plot(batch, i, ens_ids[batch_start:batch_end, :])
        # if i == 1:
        #     break
