''' The goal of this section is to analyze performance strictly over flu seasons, which leaves out summer weeks. To enable comparison with Kandula's paper,
the following itmes are produced:
1. Median and interquartile comparison of our models with benchmarks and their best results, in tabular form.
2. Violin with overlay box-whiskers plots for each model for each metric, and heatmap performance comparison with GFT.
'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

from lib import config


def get_all_seasons(table, model, metric):
    ''' table is a dataframe containing precomputed metrics. This function
    subsets the table to the specified model and metric. '''

    subtable = table[(table['Model'] == model) & (table['Metric'] == metric)]

    s14 = subtable['2014-15'].values
    s15 = subtable['2015-16'].values
    s16 = subtable['2016-17'].values

    s14 = s14[s14 != '--'].astype(np.float).reshape(-1, 1)
    s15 = s15[s15 != '--'].astype(np.float).reshape(-1, 1)
    s16 = s16[s16 != '--'].astype(np.float).reshape(-1, 1)

    return np.concatenate((s14, s15, s16))


def rank_array(table, metric, states):
    ranks = np.zeros((4,4))

    subtable = table[table['Metric'] == metric]
    for st in states:
        tmp = subtable[subtable['State'] == st]
        s14 = tmp['2014-15'].values
        s15 = tmp['2015-16'].values
        s16 = tmp['2016-17'].values

        for arr in [s14, s15, s16]:
            if arr[0] == '--':
                continue
            order = np.argsort(arr.astype(np.float))[::-1]

            if metric == 'PEARSON':
                order = order[::-1]
            for i in range(4):
                ranks[i, order[i]] += 1

    return ranks


def beat_gft_array(table, metric, states):
    ''' computes how many seasons (out of 3) each state beats GFT in, using
    the specified metric. Then aggregates this information over all states '''

    ranks = np.zeros((4,3))

    # restrict table to specified metric
    subtable = table[table['Metric'] == metric]
    for st in states:
        if st in ['AR','ID','WV']:
            continue

        # further restrict to a specific state
        tmp = subtable[subtable['State'] == st][['Model','2014-15','2015-16','2016-17']]

        # count times each model beats AR52 over the three seasons
        for i, mod in enumerate(['ARGO(gt,ath)','Net','ARGONet']):
            counter = 0
            for s in ['2014-15','2015-16','2016-17']:
                if metric == 'PEARSON':
                    counter += int(tmp[tmp['Model'] == mod][s].values >= tmp[tmp['Model'] == 'AR52'][s].values)
                else:
                    counter += int(tmp[tmp['Model'] == mod][s].values <= tmp[tmp['Model'] == 'AR52'][s].values)
            ranks[counter, i] += 1
    return ranks


def tally(table, metric, states):
    ''' counts total number of seasons where our models outperform a benchmark '''

    # restrict table to specified metric
    subtable = table[table['Metric'] == metric]

    counter = 0
    mod = 'ARGO(gt)'
    mod2 = 'ARGO(gt,ath)'

    for st in states:
        # further restrict to a specific state
        tmp = subtable[subtable['State'] == st][['Model','2014-15','2015-16','2016-17']]

        for s in ['2014-15','2015-16','2016-17']:
            if metric == 'PEARSON':
                counter += int(tmp[tmp['Model'] == mod][s].values >= tmp[tmp['Model'] == 'AR52'][s].values
                or tmp[tmp['Model'] == mod2][s].values >= tmp[tmp['Model'] == 'AR52'][s].values)
            else:
                counter += int(tmp[tmp['Model'] == mod][s].values <= tmp[tmp['Model'] == 'AR52'][s].values
                or tmp[tmp['Model'] == mod2][s].values <= tmp[tmp['Model'] == 'AR52'][s].values)
    print counter


def metrics_pie(table, model, benchmark, states):

    counts = np.zeros(4)
    for st in states:
        tmp = table[table['State'] == st]

        a = tmp[tmp['Model'] == model]['ARGONet Period'].values.astype(float) - tmp[tmp['Model'] == benchmark]['ARGONet Period'].values.astype(float)
        a[1] = -1 * a[1]

        counts[sum(a <= 0)] += 1

    return counts


def main():

    # read in table with pre-computed metrics over each season and state
    table = pd.read_csv(config.STATES_DIR + '/_overview/final_table_merged.csv')
    states = set(table.State.values)
    table = table[table.Model.isin(['AR52','ARGO(gt,ath)','Net','ARGONet'])]

    # print table.head(10)

    # # print net metrics
    # net_seasons = get_all_seasons(table, 'Net', 'PEARSON').reshape(-1, 1)
    # print 'Net:\t{0} ({1}-{2})'.format(np.percentile(net_seasons, 50),
    #                              np.percentile(net_seasons, 25),
    #                              np.percentile(net_seasons, 75))
    # net_seasons = get_all_seasons(table, 'Net', 'RMSE').reshape(-1, 1)
    # print 'Net:\t{0} ({1}-{2})'.format(np.percentile(net_seasons, 50),
    #                              np.percentile(net_seasons, 25),
    #                              np.percentile(net_seasons, 75))
    # net_seasons = get_all_seasons(table, 'Net', 'MAPE').reshape(-1, 1)
    # print 'Net:\t{0} ({1}-{2})'.format(np.percentile(net_seasons, 50),
    #                               np.percentile(net_seasons, 25),
    #                               np.percentile(net_seasons, 75))

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            if val == 0:
                return ''
            return '{v:d}'.format(p=pct,v=val)
        return my_autopct

    labels=['0/3','1/3','2/3','3/3']
    # explode = (0.03,0.03,0.03,0.03)
    explode = (0,0,0,0)

    counts1 = metrics_pie(table, 'ARGO(gt,ath)', 'AR52', states)
    counts2 = metrics_pie(table, 'Net', 'AR52', states)
    counts3 = metrics_pie(table, 'ARGONet', 'AR52', states)
    print counts1, counts2, counts3

    pie, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.pie(counts1, colors=['#e4595b','#fbc5c0','#d1e5f0','#4393c3'],
            autopct=make_autopct(counts1), pctdistance=0.75, explode=explode, startangle=0)
    ax1.axis('equal')
    ax1.set_title('ARGO vs AR52', fontsize=12)

    ax2.pie(counts2, colors=['#e4595b','#fbc5c0','#d1e5f0','#4393c3'],
            autopct=make_autopct(counts2), pctdistance=0.75, explode=explode, startangle=0)
    ax2.axis('equal')
    ax2.set_title('Net vs AR52', fontsize=12)

    patches = ax3.pie(counts3, colors=['#e4595b','#fbc5c0','#d1e5f0','#4393c3'],
            autopct=make_autopct(counts3), pctdistance=0.75, explode=explode, startangle=0)
    ax3.axis('equal')
    ax3.set_title('ARGONet vs AR52', fontsize=12)

    #draw circle
    centre_circle = plt.Circle((0,0),0.5,fc='white')
    ax1.add_artist(centre_circle)
    centre_circle = plt.Circle((0,0),0.5,fc='white')
    ax2.add_artist(centre_circle)
    centre_circle = plt.Circle((0,0),0.5,fc='white')
    ax3.add_artist(centre_circle)

    legend = ax3.legend(patches[0], labels, loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1),
                  title='Metrics outperforming', ncol=4, fontsize=12)
    legend.get_title().set_fontsize('12')

    # plt.tight_layout()
    pie.set_size_inches(4.5, 12)
    pie.savefig('pie2.png', dpi=300)
    plt.show()





    # generate dataframes subsetted to each model for a specific metric
    ar_seasons = get_all_seasons(table, 'AR52', 'PEARSON').reshape(-1, 1)
    argo1_seasons = get_all_seasons(table, 'ARGO(gt,ath)', 'PEARSON').reshape(-1, 1)
    argo2_seasons = get_all_seasons(table, 'Net', 'PEARSON').reshape(-1, 1)
    argo3_seasons = get_all_seasons(table, 'ARGONet', 'PEARSON').reshape(-1, 1)


    # combine into a new dataframe with a column for each model
    seasons = np.hstack((ar_seasons, argo1_seasons, argo2_seasons, argo3_seasons))
    seasons_pearson_df = pd.DataFrame(seasons, columns=['AR52','ARGO','Net','ARGONet'])

    # median and interquartile information
    print 'AR52:\t{0} ({1}-{2})'.format(np.percentile(ar_seasons, 50),
                                 np.percentile(ar_seasons, 25),
                                 np.percentile(ar_seasons, 75))

    print 'ARGO(gt):\t{0} ({1}-{2})'.format(np.percentile(argo1_seasons, 50),
                                 np.percentile(argo1_seasons, 25),
                                 np.percentile(argo1_seasons, 75))

    print 'ARGO(gt,ath):\t{0} ({1}-{2})'.format(np.percentile(argo2_seasons, 50),
                                 np.percentile(argo2_seasons, 25),
                                 np.percentile(argo2_seasons, 75))

    print 'ARGONet:\t{0} ({1}-{2})'.format(np.percentile(argo3_seasons, 50),
                                 np.percentile(argo3_seasons, 25),
                                 np.percentile(argo3_seasons, 75))

    # repeat for RMSE
    ar_seasons = get_all_seasons(table, 'AR52', 'RMSE').reshape(-1, 1)
    argo1_seasons = get_all_seasons(table, 'ARGO(gt,ath)', 'RMSE').reshape(-1, 1)
    argo2_seasons = get_all_seasons(table, 'Net', 'RMSE').reshape(-1, 1)
    argo3_seasons = get_all_seasons(table, 'ARGONet', 'RMSE').reshape(-1, 1)
    seasons = np.hstack((ar_seasons, argo1_seasons, argo2_seasons, argo3_seasons))
    seasons_rmse_df = pd.DataFrame(seasons, columns=['AR52','ARGO','Net','ARGONet'])


    print 'AR52:\t{0} ({1}-{2})'.format(np.percentile(ar_seasons, 50),
                                 np.percentile(ar_seasons, 25),
                                 np.percentile(ar_seasons, 75))

    print 'ARGO(gt):\t{0} ({1}-{2})'.format(np.percentile(argo1_seasons, 50),
                                 np.percentile(argo1_seasons, 25),
                                 np.percentile(argo1_seasons, 75))

    print 'ARGO(gt,ath):\t{0} ({1}-{2})'.format(np.percentile(argo2_seasons, 50),
                                 np.percentile(argo2_seasons, 25),
                                 np.percentile(argo2_seasons, 75))

    print 'ARGONet:\t{0} ({1}-{2})'.format(np.percentile(argo3_seasons, 50),
                                 np.percentile(argo3_seasons, 25),
                                 np.percentile(argo3_seasons, 75))

    # repeat for MAPE
    ar_seasons = get_all_seasons(table, 'AR52', 'MAPE').reshape(-1, 1)
    argo1_seasons = get_all_seasons(table, 'ARGO(gt,ath)', 'MAPE').reshape(-1, 1)
    argo2_seasons = get_all_seasons(table, 'Net', 'MAPE').reshape(-1, 1)
    argo3_seasons = get_all_seasons(table, 'ARGONet', 'MAPE').reshape(-1, 1)
    seasons = np.hstack((ar_seasons, argo1_seasons, argo2_seasons, argo3_seasons))
    seasons_mape_df = pd.DataFrame(seasons, columns=['AR52','ARGO','Net','ARGONet'])


    print 'AR52:\t{0} ({1}-{2})'.format(np.percentile(ar_seasons, 50),
                                 np.percentile(ar_seasons, 25),
                                 np.percentile(ar_seasons, 75))

    # print 'ARGO(gt):\t{0} ({1}-{2})'.format(np.percentile(argo1_seasons, 50),
    #                              np.percentile(argo1_seasons, 25),
    #                              np.percentile(argo1_seasons, 75))

    print 'ARGO(gt,ath):\t{0} ({1}-{2})'.format(np.percentile(argo2_seasons, 50),
                                 np.percentile(argo2_seasons, 25),
                                 np.percentile(argo2_seasons, 75))

    print 'ARGONet:\t{0} ({1}-{2})'.format(np.percentile(argo3_seasons, 50),
                                 np.percentile(argo3_seasons, 25),
                                 np.percentile(argo3_seasons, 75))

    # tally(table, 'RMSE', states)
    # tally(table, 'PEARSON', states)
    # tally(table, 'MAPE', states)

    # # generate heatmap arrays
    # rank_pearson = rank_array(table, 'PEARSON', states)
    # rank_rmse = rank_array(table, 'RMSE', states)
    # rank_mape = rank_array(table, 'MAPE', states)
    # beat_gft_pearson = beat_gft_array(table, 'PEARSON', states)
    # beat_gft_rmse = beat_gft_array(table, 'RMSE', states)
    # beat_gft_mape = beat_gft_array(table, 'MAPE', states)
    #
    # # generate subplots
    # f, axes = plt.subplots(nrows=3, ncols=3)
    #
    # # colordict
    # colordict = {'AR52': '#ffa138', 'ARGO': '#9baaff', 'ARGONet': '#e25658', 'Net':'#bfd850'}
    #
    # # make violin plots
    # sns.violinplot(data=seasons_rmse_df, ax=axes[0,0], color=1, palette=colordict, cut=0.1)
    # axes[0,0].set_title('RMSE', fontsize=12)
    # axes[0,0].set_ylim(bottom=0, top=4)
    # axes[0,0].set_yticks([0,1,2,3,4])
    #
    # sns.violinplot(data=seasons_pearson_df, ax=axes[0,1], palette=colordict, cut=0.1)
    # axes[0,1].set_title('Correlation', fontsize=12)
    # axes[0,1].set_ylim([0,1])
    # axes[0,1].set_yticks([0,0.25,0.5,0.75,1])
    #
    # sns.violinplot(data=seasons_mape_df, ax=axes[0,2], palette=colordict, cut=0.1)
    # axes[0,2].set_title('MAPE', fontsize=12)
    # axes[0,2].set_ylim([0,3])
    # axes[0,2].set_yticks([0,1,2,3])
    #
    #
    # # rank_rmse = rank_rmse[::-1,:]
    # # rank_pearson = rank_pearson[::-1,:]
    # # rank_mape = rank_mape[::-1,:]
    # #
    # # rank_rmse[2:,:] = -rank_rmse[2:,:]
    # # rank_pearson[2:,:] = -rank_pearson[2:,:]
    # # rank_mape[2:,:] = -rank_mape[2:,:]
    #
    #
    # # print rank_rmse
    # ind = np.arange(4)
    # width = 0.75
    # axes[1,0].barh(ind, rank_rmse[:,0], width, color='#ffa138', edgecolor='white')
    # axes[1,0].barh(ind, rank_rmse[:,1], width, left=rank_rmse[:,0], color='#9baaff', edgecolor='white')
    # axes[1,0].barh(ind, rank_rmse[:,2], width, left=rank_rmse[:,0] + rank_rmse[:,1], color='#bfd850', edgecolor='white')
    # axes[1,0].barh(ind, rank_rmse[:,3], width, left=rank_rmse[:,0] + rank_rmse[:,1] + rank_rmse[:,2], color='#e25658', edgecolor='white')
    #
    #
    # axes[1,1].barh(ind, rank_pearson[:,0], width, color='#ffa138', edgecolor='white')
    # axes[1,1].barh(ind, rank_pearson[:,1], width, left=rank_pearson[:,0], color='#9baaff', edgecolor='white')
    # axes[1,1].barh(ind, rank_pearson[:,2], width, left=rank_pearson[:,0] + rank_pearson[:,1], color='#bfd850', edgecolor='white')
    # axes[1,1].barh(ind, rank_pearson[:,3], width, left=rank_pearson[:,0] + rank_pearson[:,1] + rank_pearson[:,2], color='#e25658', edgecolor='white')
    #
    # axes[1,2].barh(ind, rank_mape[:,0], width, color='#ffa138', edgecolor='white')
    # axes[1,2].barh(ind, rank_mape[:,1], width, left=rank_mape[:,0], color='#9baaff', edgecolor='white')
    # axes[1,2].barh(ind, rank_mape[:,2], width, left=rank_mape[:,0] + rank_mape[:,1], color='#bfd850', edgecolor='white')
    # axes[1,2].barh(ind, rank_mape[:,3], width, left=rank_mape[:,0] + rank_mape[:,1] + rank_mape[:,2], color='#e25658', edgecolor='white')
    #
    #
    # axes[1,0].set_ylabel('Rank', fontsize=12)
    #
    # axes[1,0].set_title('RMSE', fontsize=12)
    # axes[1,1].set_title('Correlation', fontsize=12)
    # axes[1,2].set_title('MAPE', fontsize=12)
    #
    # #
    # # # make heatmaps
    # # sns.heatmap(rank_rmse, cmap='RdBu', center=0, ax=axes[1,0], cbar=False, annot=np.abs(rank_rmse))
    # # axes[1,0].set_xticklabels(['AR52','ARGO(gt)','ARGO(gt,ath)','ARGONet'])
    # # axes[1,0].set_yticklabels(['1','2','3','4'], rotation='horizontal')
    # # axes[1,0].set_ylabel('Rank')
    # #
    # # sns.heatmap(rank_pearson, cmap='RdBu', center=0, ax=axes[1,1], cbar=False, annot=np.abs(rank_pearson))
    # # axes[1,1].set_xticklabels(['AR52','ARGO(gt)','ARGO(gt,ath)','ARGONet'])
    # # axes[1,1].set_yticklabels(['1','2','3','4'], rotation='horizontal')
    # #
    # # sns.heatmap(rank_mape, cmap='RdBu', center=0, ax=axes[1,2], cbar=False, annot=np.abs(rank_mape))
    # # axes[1,2].set_xticklabels(['AR52','ARGO(gt)','ARGO(gt,ath)','ARGONet'])
    # # axes[1,2].set_yticklabels(['1','2','3','4'], rotation='horizontal')
    #
    #
    #
    # # beat_gft_rmse = beat_gft_rmse[::-1,:]
    # # beat_gft_pearson = beat_gft_pearson[::-1,:]
    # # beat_gft_mape = beat_gft_mape[::-1,:]
    #
    #
    # ind = np.arange(3)
    # width = 0.65
    # axes[2,0].bar(ind, beat_gft_rmse[0,:], width, color='#e4595b', edgecolor='white')
    # axes[2,0].bar(ind, beat_gft_rmse[1,:], width, bottom=beat_gft_rmse[0,:], color='#fbc5c0', edgecolor='white')
    # axes[2,0].bar(ind, beat_gft_rmse[2,:], width, bottom=beat_gft_rmse[0,:] + beat_gft_rmse[1,:], color='#d1e5f0', edgecolor='white')
    # axes[2,0].bar(ind, beat_gft_rmse[3,:], width, bottom=beat_gft_rmse[0,:] + beat_gft_rmse[1,:] + beat_gft_rmse[2,:], color='#4393c3', edgecolor='white')
    #
    # axes[2,1].bar(ind, beat_gft_pearson[0,:], width, color='#e4595b', edgecolor='white')
    # axes[2,1].bar(ind, beat_gft_pearson[1,:], width, bottom=beat_gft_pearson[0,:], color='#fbc5c0', edgecolor='white')
    # axes[2,1].bar(ind, beat_gft_pearson[2,:], width, bottom=beat_gft_pearson[0,:] + beat_gft_pearson[1,:], color='#d1e5f0', edgecolor='white')
    # axes[2,1].bar(ind, beat_gft_pearson[3,:], width, bottom=beat_gft_pearson[0,:] + beat_gft_pearson[1,:] + beat_gft_pearson[2,:], color='#4393c3', edgecolor='white')
    #
    # p1 = axes[2,2].bar(ind, beat_gft_mape[0,:], width, color='#e4595b', edgecolor='white')
    # p2 = axes[2,2].bar(ind, beat_gft_mape[1,:], width, bottom=beat_gft_mape[0,:], color='#fbc5c0', edgecolor='white')
    # p3 = axes[2,2].bar(ind, beat_gft_mape[2,:], width, bottom=beat_gft_mape[0,:] + beat_gft_mape[1,:], color='#d1e5f0', edgecolor='white')
    # p4 = axes[2,2].bar(ind, beat_gft_mape[3,:], width, bottom=beat_gft_mape[0,:] + beat_gft_mape[1,:] + beat_gft_mape[2,:], color='#4393c3', edgecolor='white')
    #
    #
    # axes[2,0].set_ylabel('States')
    # axes[2,0].set_title('RMSE')
    # axes[2,1].set_title('Correlation')
    # axes[2,2].set_title('MAPE')
    #
    # legend = axes[2,1].legend((p1, p2, p3, p4), ('0/3','1/3','2/3','3/3'),loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.2),
    #                   title='Seasons outperfoming AR52', ncol=4, fontsize=12)
    # legend.get_title().set_fontsize('12')
    # # # make heatmaps
    # # sns.heatmap(beat_gft_rmse, vmin=-1, cmap='Purples', ax=axes[2,0], cbar=False, annot=True)
    # # axes[2,0].set_xticklabels(['ARGO(gt)','ARGO(gt,ath)','ARGONet'])
    # # axes[2,0].set_yticklabels(['3','2','1','0'], rotation='horizontal')
    # # axes[2,0].set_ylabel('Seasons outperforming AR52')
    # #
    # # sns.heatmap(beat_gft_pearson, vmin=-1, cmap='Purples', ax=axes[2,1], cbar=False, annot=True)
    # # axes[2,1].set_xticklabels(['ARGO(gt)','ARGO(gt,ath)','ARGONet'])
    # # axes[2,1].set_yticklabels(['3','2','1','0'], rotation='horizontal')
    # #
    # # sns.heatmap(beat_gft_mape, vmin=-1, cmap='Purples', ax=axes[2,2], cbar=False, annot=True)
    # # axes[2,2].set_xticklabels(['ARGO(gt)','ARGO(gt,ath)','ARGONet'])
    # # axes[2,2].set_yticklabels(['3','2','1','0'], rotation='horizontal')
    #
    #
    # for ax in [axes[0,0], axes[0,1], axes[0,2]]:
    #
    #     ax.set_xticklabels(['AR52','ARGO','Net','ARGONet'])
    #
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_linewidth(1.5)
    #
    #     ax.tick_params(axis='y', rotation=0, length=6, width=1.5, labelsize=12)
    #     ax.tick_params(axis='x', length=0, width=1.5, labelsize=12)
    #
    # for ax in [axes[1,0], axes[1,1], axes[1,2]]:
    #
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_linewidth(1.5)
    #
    #     ax.tick_params(axis='x', rotation=0, length=6, width=1.5, labelsize=12)
    #     ax.tick_params(axis='y', length=0, width=1.5, labelsize=12)
    #
    #     ax.set_ylim([-0.5,3.5])
    #     ax.set_yticks([0,1,2,3])
    #     ax.set_yticklabels(['4th','3rd','2nd','1st'])
    #     ax.set_xlabel('Seasons', fontsize=12)
    #     ax.spines['bottom'].set_bounds(0, 108)
    #     ax.set_xlim([0,108])
    #     ax.set_xticks([0, 20, 40, 60, 80, 100, 108])
    #     ax.set_xticklabels([0, 20, 40, 60, 80, '', 108])
    #
    # for ax in [axes[2,0], axes[2,1], axes[2,2]]:
    #
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_linewidth(1.5)
    #     ax.spines['left'].set_bounds(0, 34)
    #
    #
    #     ax.tick_params(axis='x', rotation=0, length=0, width=1.5, labelsize=12)
    #     ax.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    #
    #     ax.set_ylim([0,35])
    #     ax.set_yticks([0,10,20,30,34])
    #     # ax.set_yticklabels(['4th','3rd','2nd','1st'])
    #     # ax.set_xlabel('Seasons')
    #     # ax.spines['bottom'].set_bounds(0, 108)
    #     ax.set_xlim([-0.5, 2.5])
    #     ax.set_xticks([0, 1, 2])
    #     ax.set_xticklabels(['ARGO','Net','ARGONet'])
    #
    # plt.subplots_adjust(hspace=0.35)


    # f.set_size_inches(13.5, 10.5)
    # f.savefig('seasons2.png', dpi=300)
    # # plt.show()


if __name__ == '__main__':
    main()
