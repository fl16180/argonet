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

STATES_PER_PAGE = 6

n_x = 2
n_y = 3

def main_plot(STATES, page):

    fig = plt.figure()

    for i, state in enumerate(STATES):
        ax = plt.subplot(n_y, n_x, i + 1)

        models = pd.read_csv(config.STATES_DIR + '/{0}/top_argo_preds.csv'.format(state), parse_dates=[0])

        models = models.iloc[92:,:]
        weeks = models['Week']
        models = models.set_index('Week')

        # index_biweek = pd.date_range(weeks.values[0], weeks.values[-1], freq='14D')
        # models1 = models.copy()
        # models = models.reindex(index=index_biweek).interpolate('linear')
        # print models1.head()
        # print models.head()
        # df_smooth = df_smooth.rename(columns={'value':'smooth'})

        weeks = models.index.values
        target = models['ILI']
        gft = models['GFT']
        argo1 = models['ARGO']

        argo1[target == 0] = np.nan
        gft[target == 0] = np.nan
        gft[gft == 0] = np.nan
        target[target == 0] = np.nan


        b, = ax.plot_date(x=weeks, y=gft, fmt="--", color='green', label='GFT', linewidth=2.2, dashes=(5, 3))
        # c, = ax.plot_date(x=weeks, y=argo1, fmt="-", color='cornflowerblue', label='ARGO(gt)', linewidth=5)
        a, = ax.plot_date(x=weeks, y=target, fmt="k-", label=r'% ILI', linewidth=3)
        d, = ax.plot_date(x=weeks, y=argo1, fmt="--", color='orangered', label='ARGO', linewidth=3, dashes=(5, 4))

        if state=='GA':
            ax.legend(handles=[a,b,d], loc='upper right', frameon=False, fontsize=13)

        ax.xaxis.set_major_locator(mdates.YearLocator())

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(2)
        # ax.Axes.tick_params()

        if state in ['MI','NC','OH','PA','SC','TN','VA']:
            ax.text(.53,0.86,state,
                horizontalalignment='center',
                fontsize=18,
                transform=ax.transAxes)
        else:
            ax.text(.50,0.86,state,
                horizontalalignment='center',
                fontsize=18,
                transform=ax.transAxes)

        ax.tick_params(axis='y', rotation=90, length=8, width=2)
        ax.tick_params(axis='x', length=6, width=2)
        ax.locator_params(axis='y', tight=True, nbins=4)


        if i < n_x * (n_y - 1):
            ax.axes.xaxis.set_ticklabels([])
        if np.mod(i, n_x) == 0:
            ax.set_ylabel('%ILI', fontsize=14)

        txs = ax.yaxis.get_majorticklocs()
        if state in ['OR','SC']:
            ax.spines['left'].set_bounds(0, 7.5)
        else:
            ax.spines['left'].set_bounds(0, txs[-2])

        txs2 = ax.xaxis.get_majorticklocs()
        ax.spines['bottom'].set_bounds(734869, 736330)
        ax.set_xlim((734691.65, 736547.35))
        hspace = np.abs(txs[1] - txs[0]) / 10
        ax.set_ylim(bottom=-hspace)


    # plt.legend(loc='best', handles=[a,b,c,d])

    plt.subplots_adjust(wspace=0.15, hspace=0.21)


    fig = plt.gcf()
    fig.set_size_inches(16, 15)
    # fig.savefig(config.STATES_DIR + '/_overview/all_states_{0}.png'.format(page), format='png', dpi=200)

    plt.show()

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
