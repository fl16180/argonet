import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from lib import config

plt.style.use('seaborn-poster')


VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']

STATES_PER_PAGE = 40

n_x = 5
n_y = 8


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


def main_plot(STATES, page):

    fig = plt.figure()

    for i, state in enumerate(STATES):
        ax = plt.subplot(n_y, n_x, i + 1)

        models = pd.read_csv(config.STATES_DIR + '/{0}/top_argo_preds.csv'.format(state), parse_dates=[0])
        ens_models = pd.read_csv(config.STATES_DIR + '/{0}/argo_net_ens_preds.csv'.format(state), parse_dates=[0])

        # load first set of models
        models = models.iloc[92:,:]
        weeks = models['Week']
        models = models.set_index('Week')
        ens_models = ens_models.set_index('Week')


        weeks = models.index.values
        target = models['ILI']
        gft = models['GFT']
        argo = models['ARGO']

        gft[target == 0] = np.nan
        gft[gft == 0] = np.nan
        argo[target == 0] = np.nan
        target[target == 0] = np.nan

        # load ensemble model
        weeks_ens = ens_models.index.values
        target_ens = ens_models['ILI']
        ens = ens_models['ARGONet']
        ens[target_ens == 0] = np.nan


        target = smooth(target, window_len=6, window='hanning')
        argo = smooth(argo, window_len=6, window='hanning')
        ens = smooth(ens, window_len=6, window='hanning')
        gft = smooth(gft, window_len=6, window='hanning')


        b, = ax.plot_date(x=weeks, y=gft, fmt="-", color='#4f9907', label='GFT', linewidth=1, alpha=0.6)
        c, = ax.plot_date(x=weeks, y=argo, fmt="-", color='#8196d3', label='ARGO', linewidth=2.5, alpha=0.85)
        a, = ax.plot_date(x=weeks, y=target, fmt="k-", label=r'%ILI', linewidth=1.5, alpha=0.8)
        d, = ax.plot_date(x=weeks_ens, y=ens, fmt="-", color='#e25658', label='ARGONet', linewidth=2.5, dashes=(5, 5))

        if state in ['VT']:
            ax.legend(handles=[a,b,c,d], loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.5), ncol=4)

        ax.xaxis.set_major_locator(mdates.YearLocator())

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(1.5)
        # ax.Axes.tick_params()

        if state in ['MI','OH','VA']:
            ax.text(.56,0.87,state,
                horizontalalignment='center',
                fontsize=16,
                transform=ax.transAxes)
        elif state in ['NC','PA','SC']:
            ax.text(.50,0.87,state,
                horizontalalignment='center',
                fontsize=16,
                transform=ax.transAxes)
        else:
            ax.text(.50,0.86,state,
                horizontalalignment='center',
                fontsize=16,
                transform=ax.transAxes)

        ax.tick_params(axis='y', rotation=90, length=5, width=1.5, labelsize=11, pad=4)
        ax.tick_params(axis='x', length=5, width=1.5, labelsize=11)
        ax.locator_params(axis='y', tight=True, nbins=4)


        # if i < n_x * (n_y - 1):
        #     ax.axes.xaxis.set_ticklabels([])
        if np.mod(i, n_x) == 0:
            ax.set_ylabel('%ILI', fontsize=12, labelpad=8)
        if state not in ['WI','WV','VA','VT','WA']:
            ax.axes.xaxis.set_ticklabels([])


        txs = ax.yaxis.get_majorticklocs()
        if state in ['OR','SC', 'VA']:
            ax.spines['left'].set_bounds(0, 7.5)
        else:
            ax.spines['left'].set_bounds(0, txs[-2])

        txs2 = ax.xaxis.get_majorticklocs()
        ax.spines['bottom'].set_bounds(734869, 736330)
        ax.set_xlim((734691.65, 736547.35))
        hspace = np.abs(txs[1] - txs[0]) / 10
        ax.set_ylim(bottom=-hspace)

        if state in ['VA']:
            ax.set_ylim(bottom=-hspace, top=9)



    # plt.legend(loc='best', handles=[a,b,c,d])

    plt.subplots_adjust(wspace=0.15, hspace=0.18)


    fig = plt.gcf()
    fig.set_size_inches(18, 20)
    fig.savefig(config.STATES_DIR + '/_overview/full_states_{0}.png'.format(page), format='png', dpi=300)

    # plt.show()

#
# def sns_plot(STATES, page):
#
#     fig = plt.figure()
#
#     for i, state in enumerate(STATES):
#         ax = plt.subplot(3, 2, i + 1)
#
#         models = pd.read_csv(config.STATES_DIR + '/{0}/top_preds.csv'.format(state), parse_dates=[0])
#
#         weeks = models['Week'][92:]
#         target = models['ILI'][92:]
#         gft = models['GFT'][92:]
#         argo1 = models['ARGO(gt)'][92:]
#         argo2 = models['ARGO(gt,ath)'][92:]
#
#         gft[gft == 0] = np.nan
#         target[target == 0] = np.nan
#
#
#         a, = ax.plot_date(x=weeks, y=target, fmt="k-", label=r'% ILI', linewidth=3.5)
#         b, = ax.plot_date(x=weeks, y=gft, fmt="-", color='green', label='GFT', linewidth=2)
#         c, = ax.plot_date(x=weeks, y=argo1, fmt="-", color='blue', label='ARGO(gt)', linewidth=2)
#         d, = ax.plot_date(x=weeks, y=argo2, fmt="-", color='red', label='ARGO(gt,ath)', linewidth=2)
#
#         ax.xaxis.set_major_locator(mdates.YearLocator())
#
#         if i < 4:
#             ax.axes.xaxis.set_ticklabels([])
#
#
#         ax.text(.52,.9,state,
#             horizontalalignment='center',
#             fontsize=18,
#             transform=ax.transAxes)
#         # plt.title(state)
#
#     # last = plt.subplot(3, 3, 9)
#     # last.set_frame_on(False)
#     # last.get_xaxis().set_visible(False)
#     # last.get_yaxis().set_visible(False)
#     # last.legend(handles=[a, b, c, d])
#
#     plt.legend(loc='best', handles=[a,b,c,d])
#
#
#     fig.text(0.012, 0.5, '% ILI', va='center', rotation='vertical', fontsize='20')
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#     # plt.tight_layout()
#
#     fig = plt.gcf()
#     fig.set_size_inches(22, 18)
#     fig.savefig(config.STATES_DIR + '/_overview/all_states_{0}.png'.format(page), format='png', dpi=200)
#
#     #plt.show()




if __name__ == '__main__':

    batch_intervals = range(0, len(VALID_STATES), STATES_PER_PAGE)

    for i, batch_start in enumerate(batch_intervals):
        batch_end = min(batch_start + STATES_PER_PAGE, len(VALID_STATES))
        batch = VALID_STATES[batch_start:batch_end]

        print i
        main_plot(batch, i)
