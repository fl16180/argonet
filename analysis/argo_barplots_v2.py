''' Generates bar plots corresponding to Figures 1 and 2 in the manuscript. These are:

Fig. 1) Comparisons over the states, showing GFT, ARGO(gt), and ARGO(gt,ath). These are scored over
the GFT period. There are 3x1 subplots showing RMSE, correlation, and MAPE.
    Note: Furthermore, the first two panels have states ordered by difference between
        ARGO(gt) and GFT in RMSE. The last panel is ordered by difference in MAPE.

Fig. 2) Comparisons over the states, showing AR52, ARGO(gt), and ARGO(gt,ath). These are scored over
the entire period. There are 3x1 subplots showing RMSE, correlation, and MAPE.
    Note: Same ordering.

Metric results are loaded from the master table "final_table.csv".
'''

from __future__ import division
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import config

# plt.rcParams["font.family"] = "consolas"

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def main():

    # read in table with pre-computed metrics over each season and state
    table = pd.read_csv(config.STATES_DIR + '/_overview/final_table.csv')
    states = list(set(table.State.values))

    # limit to specific metrics
    table_rmse = table[table.Metric == 'RMSE']
    table_corr = table[table.Metric == 'PEARSON']
    table_mape = table[table.Metric == 'MAPE']


    ####################################################
    ############         Figure 1           ############
    ####################################################

    # get list of differences between RMSEs
    ardiff1 = [(table_rmse[(table_rmse.Model == 'AR52') & (table_rmse.State == st)]['GFT Period'].values[0] \
            - table_rmse[(table_rmse.Model == 'ARGO(gt,ath)') & (table_rmse.State == st)]['GFT Period'].values[0])
            for st in states]
    gftdiff1 = [(table_rmse[(table_rmse.Model == 'GFT') & (table_rmse.State == st)]['GFT Period'].values[0] \
            - table_rmse[(table_rmse.Model == 'ARGO(gt,ath)') & (table_rmse.State == st)]['GFT Period'].values[0])
            for st in states]

    ardiff2 = [(table_corr[(table_corr.Model == 'AR52') & (table_corr.State == st)]['GFT Period'].values[0] \
            - table_corr[(table_corr.Model == 'ARGO(gt,ath)') & (table_corr.State == st)]['GFT Period'].values[0])
            for st in states]
    gftdiff2 = [(table_corr[(table_corr.Model == 'GFT') & (table_corr.State == st)]['GFT Period'].values[0] \
            - table_corr[(table_corr.Model == 'ARGO(gt,ath)') & (table_corr.State == st)]['GFT Period'].values[0])
            for st in states]

    ardiff3 = [(table_mape[(table_mape.Model == 'AR52') & (table_mape.State == st)]['GFT Period'].values[0] \
            - table_mape[(table_mape.Model == 'ARGO(gt,ath)') & (table_mape.State == st)]['GFT Period'].values[0])
            for st in states]
    gftdiff3 = [(table_mape[(table_mape.Model == 'GFT') & (table_mape.State == st)]['GFT Period'].values[0] \
            - table_mape[(table_mape.Model == 'ARGO(gt,ath)') & (table_mape.State == st)]['GFT Period'].values[0])
            for st in states]

    rmse_diffs = [(1000 * (i > 0.03) + 100 * ((i > 0.01) & (i <= 0.03)) + 10 * ((i > -0.02) & (i <= 0.01))) * j for i, j in zip(ardiff1, gftdiff1)]
    corr_diffs = [(1000 * (i > 0.03) + 100 * ((i > 0.01) & (i <= 0.03)) + 10 * ((i > -0.02) & (i <= 0.01))) * j - 5000 * (j < 0) for i, j in zip(ardiff2, gftdiff2)]
    mape_diffs = [(1000 * (i > 0.03) + 100 * ((i > 0.01) & (i <= 0.03)) + 10 * ((i > -0.02) & (i <= 0.01))) * j - 5000 * (j < 0) for i, j in zip(ardiff3, gftdiff3)]



    # reorder states with respect to the ordered diffs
    states_rmse_order = [states[i] for i in argsort(rmse_diffs)[::-1]]
    states_corr_order = [states[i] for i in argsort(corr_diffs)]
    states_mape_order = [states[i] for i in argsort(mape_diffs)[::-1]]


    ### GENERATE ARRAYS FOR BAR PLOTS ###

    # ARRAYS FOR RMSE (PANEL 1)
    gft_rmse = []
    for st in states_rmse_order:
        tmp = table[(table.Metric == 'RMSE') & (table.Model == 'GFT') & (table.State == st)]['GFT Period'].values[0]
        gft_rmse.append(tmp)
    argt_rmse = []
    for st in states_rmse_order:
        tmp = table[(table.Metric == 'RMSE') & (table.Model == 'AR52') & (table.State == st)]['GFT Period'].values[0]
        argt_rmse.append(tmp)
    argo_rmse = []
    for st in states_rmse_order:
        tmp = table[(table.Metric == 'RMSE') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['GFT Period'].values[0]
        argo_rmse.append(tmp)
    # END BLOCK

    # ARRAYS FOR CORRELATION (PANEL 2)
    gft_corr = []
    for st in states_corr_order:
        tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'GFT') & (table.State == st)]['GFT Period'].values[0]
        gft_corr.append(tmp)
    argt_corr = []
    for st in states_corr_order:
        tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'AR52') & (table.State == st)]['GFT Period'].values[0]
        argt_corr.append(tmp)
    argo_corr = []
    for st in states_corr_order:
        tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['GFT Period'].values[0]
        argo_corr.append(tmp)
    # END BLOCK

    # ARRAYS FOR MAPE (PANEL 3)
    gft_mape = []
    for st in states_mape_order:
        tmp = table[(table.Metric == 'MAPE') & (table.Model == 'GFT') & (table.State == st)]['GFT Period'].values[0]
        gft_mape.append(tmp)
    argt_mape = []
    for st in states_mape_order:
        tmp = table[(table.Metric == 'MAPE') & (table.Model == 'AR52') & (table.State == st)]['GFT Period'].values[0]
        argt_mape.append(tmp)
    argo_mape = []
    for st in states_mape_order:
        tmp = table[(table.Metric == 'MAPE') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['GFT Period'].values[0]
        argo_mape.append(tmp)
    # END BLOCK


    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # divider = make_axes_locatable(ax1)
    # ax_new = divider.new_vertical(size="100%", pad=0.1)
    # f.add_axes(ax_new)

    ind = np.arange(len(states))
    width = 0.22

    p1 = ax1.bar(ind - width, gft_rmse, width, color='#4f9907', edgecolor='k', linewidth=0.25)
    p2 = ax1.bar(ind, argt_rmse, width, color='#ffa138', edgecolor='k', linewidth=0.25)
    p3 = ax1.bar(ind + width, argo_rmse, width, color='#9baaff', alpha=0.95, edgecolor='k', linewidth=0.25)

    ax1.set_xticks(ind)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.set_xticklabels(states_rmse_order)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_axisbelow(True)
    # ax1.grid(color='k', axis='y', linestyle='-', linewidth=1, alpha=0.1)
    ax1.set_xlim([-1,37])
    ax1.set_ylim([0,3.2])
    ax1.set_yticks([0,1.6,3.2])
    ax1.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    ax1.tick_params(axis='x', length=0, width=1.5, labelsize=11)
    ax1.locator_params(axis='y', tight=True, nbins=5)

    # d = .015
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    # ax1.set_yscale('log')

    # ax_new.bar(ind - width, gft_rmse, width, color='#4f9907', edgecolor='k', linewidth=0.5)
    # ax_new.bar(ind, argt_rmse, width, color='#9baaff', edgecolor='k', linewidth=0.5)
    # ax_new.bar(ind + width, argo_rmse, width, color='#ffa138', alpha=0.95, edgecolor='k', linewidth=0.5)
    # ax_new.set_ylim([4, 6])
    # ax_new.tick_params(bottom="off", labelbottom='off')
    # ax_new.spines['bottom'].set_visible(False)


    p4 = ax2.bar(ind - width, gft_corr, width, color='#4f9907', edgecolor='k', linewidth=0.25)
    p5 = ax2.bar(ind, argt_corr, width, color='#ffa138', edgecolor='k', linewidth=0.25)
    p6 = ax2.bar(ind + width, argo_corr, width, color='#9baaff', alpha=0.95, edgecolor='k', linewidth=0.25)

    ax2.set_xticks(ind)
    ax2.set_xticklabels(states_corr_order)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_axisbelow(True)
    # ax2.grid(color='k', axis='y', linestyle='-', linewidth=1, alpha=0.1)
    ax2.set_xlim([-1,37])
    ax2.set_ylim([0.4,1])
    ax2.set_yticks([0.4,0.7,1])
    ax2.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    ax2.tick_params(axis='x', length=0, width=1.5, labelsize=11)
    ax2.locator_params(axis='y', tight=True, nbins=4)


    p7 = ax3.bar(ind - width, gft_mape, width, color='#4f9907', edgecolor='k', linewidth=0.25)
    p8 = ax3.bar(ind, argt_mape, width, color='#ffa138', edgecolor='k', linewidth=0.25)
    p9 = ax3.bar(ind + width, argo_mape, width, color='#9baaff', alpha=0.95, edgecolor='k', linewidth=0.25)

    ax3.set_xticks(ind)
    ax3.set_xticklabels(states_mape_order)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.set_ylabel('MAPE', fontsize=12)
    ax3.set_axisbelow(True)
    # ax3.grid(color='k', axis='y', linestyle='-', linewidth=1, alpha=0.1)
    ax3.set_xlim([-1,37])
    ax3.set_ylim([0,2.2])
    ax3.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    ax3.tick_params(axis='x', length=0, width=1.5, labelsize=11)
    ax3.set_yticks([0,1.1,2.2])
    ax3.locator_params(axis='y', tight=True)

    for axis in ['bottom','left']:
        ax1.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_linewidth(1.5)
        ax3.spines[axis].set_linewidth(1.5)

    ax1.get_yaxis().set_label_coords(-0.045,0.5)
    ax2.get_yaxis().set_label_coords(-0.045,0.5)
    ax3.get_yaxis().set_label_coords(-0.045,0.5)

    ax3.legend((p1[0], p2[0], p3[0]), ('GFT','AR52', 'ARGO'), loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12)

    f.set_size_inches(16, 8)
    f.savefig('barplot1.png', dpi=300)
    plt.show()
    #



    ###################################################
    ###########         Figure 2           ############
    ###################################################

    # # # get list of differences between RMSEs
    # # rmse_diffs = [table_rmse[(table_rmse.Model == 'AR52') & (table_rmse.State == st)]['ARGONet Period'].values[0] \
    # #         - table_rmse[(table_rmse.Model == 'ARGONet') & (table_rmse.State == st)]['ARGONet Period'].values[0]
    # #         for st in states]
    # #
    # # # get list of differences between correlations
    # # corr_diffs = [table_corr[(table_corr.Model == 'AR52') & (table_corr.State == st)]['ARGONet Period'].values[0] \
    # #         - table_corr[(table_corr.Model == 'ARGONet') & (table_corr.State == st)]['ARGONet Period'].values[0]
    # #         for st in states]
    # #
    # # # get list of differences between MAPEs
    # # mape_diffs = [table_mape[(table_mape.Model == 'AR52') & (table_mape.State == st)]['ARGONet Period'].values[0] \
    # #     - table_mape[(table_mape.Model == 'ARGONet') & (table_mape.State == st)]['ARGONet Period'].values[0]
    # #     for st in states]
    #
    # # get list of differences between RMSEs
    # ardiff1 = [(table_rmse[(table_rmse.Model == 'AR52') & (table_rmse.State == st)]['ARGONet Period'].values[0] \
    #         - table_rmse[(table_rmse.Model == 'ARGONet') & (table_rmse.State == st)]['ARGONet Period'].values[0])
    #         for st in states]
    # gftdiff1 = [(table_rmse[(table_rmse.Model == 'ARGO(gt,ath)') & (table_rmse.State == st)]['ARGONet Period'].values[0] \
    #         - table_rmse[(table_rmse.Model == 'ARGONet') & (table_rmse.State == st)]['ARGONet Period'].values[0])
    #         for st in states]
    #
    # ardiff2 = [(table_corr[(table_corr.Model == 'AR52') & (table_corr.State == st)]['ARGONet Period'].values[0] \
    #         - table_corr[(table_corr.Model == 'ARGONet') & (table_corr.State == st)]['ARGONet Period'].values[0])
    #         for st in states]
    # gftdiff2 = [(table_corr[(table_corr.Model == 'ARGO(gt,ath)') & (table_corr.State == st)]['ARGONet Period'].values[0] \
    #         - table_corr[(table_corr.Model == 'ARGONet') & (table_corr.State == st)]['ARGONet Period'].values[0])
    #         for st in states]
    #
    # ardiff3 = [(table_mape[(table_mape.Model == 'AR52') & (table_mape.State == st)]['ARGONet Period'].values[0] \
    #         - table_mape[(table_mape.Model == 'ARGONet') & (table_mape.State == st)]['ARGONet Period'].values[0])
    #         for st in states]
    # gftdiff3 = [(table_mape[(table_mape.Model == 'ARGO(gt,ath)') & (table_mape.State == st)]['ARGONet Period'].values[0] \
    #         - table_mape[(table_mape.Model == 'ARGONet') & (table_mape.State == st)]['ARGONet Period'].values[0])
    #         for st in states]
    #
    # rmse_diffs = [(1000 * (i > 0.03) + 100 * ((i > 0.01) & (i <= 0.03)) + 10 * ((i > -0.01) & (i <= 0.01))) * j for i, j in zip(ardiff1, gftdiff1)]
    # corr_diffs = [(1000 * (i > 0.03) + 100 * ((i > 0.01) & (i <= 0.03)) + 10 * ((i > -0.01) & (i <= 0.01))) * j - 5000 * (j < 0) for i, j in zip(ardiff2, gftdiff2)]
    # mape_diffs = [(1000 * (i > 0.03) + 100 * ((i > 0.01) & (i <= 0.03)) + 10 * ((i > -0.01) & (i <= 0.01))) * j - 5000 * (j < 0) for i, j in zip(ardiff3, gftdiff3)]
    #
    #
    #
    # # reorder states with respect to the ordered diffs
    # states_rmse_order = [states[i] for i in argsort(rmse_diffs)[::-1]]
    # states_corr_order = [states[i] for i in argsort(corr_diffs)]
    # states_mape_order = [states[i] for i in argsort(mape_diffs)[::-1]]
    #
    #
    # ### GENERATE ARRAYS FOR BAR PLOTS ###
    #
    # # ARRAYS FOR RMSE (PANEL 1)
    # ar_rmse = []
    # for st in states_rmse_order:
    #     tmp = table[(table.Metric == 'RMSE') & (table.Model == 'AR52') & (table.State == st)]['ARGONet Period'].values[0]
    #     ar_rmse.append(tmp)
    # argo_rmse = []
    # for st in states_rmse_order:
    #     tmp = table[(table.Metric == 'RMSE') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['ARGONet Period'].values[0]
    #     argo_rmse.append(tmp)
    # ens_rmse = []
    # for st in states_rmse_order:
    #     tmp = table[(table.Metric == 'RMSE') & (table.Model == 'ARGONet') & (table.State == st)]['ARGONet Period'].values[0]
    #     ens_rmse.append(tmp)
    # # END BLOCK
    #
    # # ARRAYS FOR CORRELATION (PANEL 2)
    # ar_corr = []
    # for st in states_corr_order:
    #     tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'AR52') & (table.State == st)]['ARGONet Period'].values[0]
    #     ar_corr.append(tmp)
    # argo_corr = []
    # for st in states_corr_order:
    #     tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['ARGONet Period'].values[0]
    #     argo_corr.append(tmp)
    # ens_corr = []
    # for st in states_corr_order:
    #     tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'ARGONet') & (table.State == st)]['ARGONet Period'].values[0]
    #     ens_corr.append(tmp)
    # # END BLOCK
    #
    # # ARRAYS FOR MAPE (PANEL 3)
    # ar_mape = []
    # for st in states_mape_order:
    #     tmp = table[(table.Metric == 'MAPE') & (table.Model == 'AR52') & (table.State == st)]['ARGONet Period'].values[0]
    #     ar_mape.append(tmp)
    # argo_mape = []
    # for st in states_mape_order:
    #     tmp = table[(table.Metric == 'MAPE') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['ARGONet Period'].values[0]
    #     argo_mape.append(tmp)
    # ens_mape = []
    # for st in states_mape_order:
    #     tmp = table[(table.Metric == 'MAPE') & (table.Model == 'ARGONet') & (table.State == st)]['ARGONet Period'].values[0]
    #     ens_mape.append(tmp)
    # # END BLOCK
    #
    #
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # # divider = make_axes_locatable(ax1)
    # # ax_new = divider.new_vertical(size="100%", pad=0.1)
    # # f.add_axes(ax_new)
    #
    # ind = np.arange(len(states))
    # width = 0.22
    #
    # p1 = ax1.bar(ind - width, ar_rmse, width, color='#ffa138', edgecolor='k', linewidth=0.25)
    # p2 = ax1.bar(ind, argo_rmse, width, color='#9baaff', edgecolor='k', linewidth=0.25)
    # p3 = ax1.bar(ind + width, ens_rmse, width, color='#e25658', alpha=0.95, edgecolor='k', linewidth=0.25)
    #
    # ax1.set_xticks(ind)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    #
    # ax1.set_xticklabels(states_rmse_order)
    # ax1.set_ylabel('RMSE', fontsize=12)
    # ax1.set_axisbelow(True)
    # # ax1.grid(color='k', axis='y', linestyle='-', linewidth=1, alpha=0.1)
    # ax1.set_xlim([-1,37])
    # ax1.set_ylim([0,2.8])
    # ax1.set_yticks([0,1.4,2.8])
    # ax1.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    # ax1.tick_params(axis='x', length=0, width=1.5, rotation=0, labelsize=11)
    # ax1.locator_params(axis='y', tight=True, nbins=5)
    #
    # # d = .015
    # # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # # ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    # # ax1.set_yscale('log')
    #
    # # ax_new.bar(ind - width, gft_rmse, width, color='#4f9907', edgecolor='k', linewidth=0.5)
    # # ax_new.bar(ind, argt_rmse, width, color='#9baaff', edgecolor='k', linewidth=0.5)
    # # ax_new.bar(ind + width, argo_rmse, width, color='#ffa138', alpha=0.95, edgecolor='k', linewidth=0.5)
    # # ax_new.set_ylim([4, 6])
    # # ax_new.tick_params(bottom="off", labelbottom='off')
    # # ax_new.spines['bottom'].set_visible(False)
    #
    #
    # p4 = ax2.bar(ind - width, ar_corr, width, color='#ffa138', edgecolor='k', linewidth=0.25)
    # p5 = ax2.bar(ind, argo_corr, width, color='#9baaff', edgecolor='k', linewidth=0.25)
    # p6 = ax2.bar(ind + width, ens_corr, width, color='#e25658', alpha=0.95, edgecolor='k', linewidth=0.25)
    #
    # ax2.set_xticks(ind)
    # ax2.set_xticklabels(states_corr_order)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    #
    # ax2.set_ylabel('Correlation', fontsize=12)
    # ax2.set_axisbelow(True)
    # # ax2.grid(color='k', axis='y', linestyle='-', linewidth=1, alpha=0.1)
    # ax2.set_xlim([-1,37])
    # ax2.set_ylim([.6,1])
    # ax2.set_yticks([.6,0.8,1])
    # ax2.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    # ax2.tick_params(axis='x', length=0, width=1.5, rotation=0, labelsize=11)
    # ax2.locator_params(axis='y', tight=True, nbins=4)
    #
    #
    # p7 = ax3.bar(ind - width, ar_mape, width, color='#ffa138', edgecolor='k', linewidth=0.25)
    # p8 = ax3.bar(ind, argo_mape, width, color='#9baaff', edgecolor='k', linewidth=0.25)
    # p9 = ax3.bar(ind + width, ens_mape, width, color='#e25658', alpha=0.95, edgecolor='k', linewidth=0.25)
    #
    # ax3.set_xticks(ind)
    # ax3.set_xticklabels(states_mape_order)
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['bottom'].set_visible(False)
    #
    # ax3.set_ylabel('MAPE', fontsize=12)
    # ax3.set_axisbelow(True)
    # # ax3.grid(color='k', axis='y', linestyle='-', linewidth=1, alpha=0.1)
    # ax3.set_xlim([-1,37])
    # ax3.set_ylim([0,2.2])
    # ax3.tick_params(axis='y', length=6, width=1.5, labelsize=12)
    # ax3.tick_params(axis='x', length=0, width=1.5, rotation=0, labelsize=11)
    # ax3.set_yticks([0,1.1,2.2])
    # ax3.locator_params(axis='y', tight=True)
    #
    # for axis in ['bottom','left']:
    #     ax1.spines[axis].set_linewidth(1.5)
    #     ax2.spines[axis].set_linewidth(1.5)
    #     ax3.spines[axis].set_linewidth(1.5)
    #
    # ax1.get_yaxis().set_label_coords(-0.045,0.5)
    # ax2.get_yaxis().set_label_coords(-0.045,0.5)
    # ax3.get_yaxis().set_label_coords(-0.045,0.5)
    #
    # ax3.legend((p1[0], p2[0], p3[0]), ('AR52','ARGO', 'ARGONet'), loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12)
    #
    # f.set_size_inches(16, 8)
    # f.savefig('barplot2.png', dpi=300)
    # plt.show()


    ####################################################
    ############         Figure 3           ############
    ####################################################

    # # get list of differences between RMSEs
    # rmse_diffs = [table_rmse[(table_rmse.Model == 'ARGO(gt,ath)') & (table_rmse.State == st)]['ARGONet Period'].values[0] \
    #         / table_rmse[(table_rmse.Model == 'ARGONet') & (table_rmse.State == st)]['ARGONet Period'].values[0]
    #         for st in states]
    #
    # # get list of differences between correlations
    # corr_diffs = [table_corr[(table_corr.Model == 'ARGO(gt,ath)') & (table_corr.State == st)]['ARGONet Period'].values[0] \
    #         / table_corr[(table_corr.Model == 'ARGONet') & (table_corr.State == st)]['ARGONet Period'].values[0]
    #         for st in states]
    #
    # # get list of differences between MAPEs
    # mape_diffs = [table_mape[(table_mape.Model == 'ARGO(gt,ath)') & (table_mape.State == st)]['ARGONet Period'].values[0] \
    #     / table_mape[(table_mape.Model == 'ARGONet') & (table_mape.State == st)]['ARGONet Period'].values[0]
    #     for st in states]
    #
    # # reorder states with respect to the ordered diffs
    # states_rmse_order = [states[i] for i in argsort(rmse_diffs)[::-1]]
    # states_corr_order = [states[i] for i in argsort(corr_diffs)]
    # states_mape_order = [states[i] for i in argsort(mape_diffs)[::-1]]
    #
    # ### GENERATE ARRAYS FOR BAR PLOTS ###
    #
    # # ARRAYS FOR RMSE (PANEL 1)
    # ar_rmse = []
    # for st in states_rmse_order:
    #     tmp = table[(table.Metric == 'RMSE') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['ARGONet Period'].values[0]
    #     ar_rmse.append(tmp)
    # argt_rmse = []
    # for st in states_rmse_order:
    #     tmp = table[(table.Metric == 'RMSE') & (table.Model == 'Net') & (table.State == st)]['ARGONet Period'].values[0]
    #     argt_rmse.append(tmp)
    # argo_rmse = []
    # for st in states_rmse_order:
    #     tmp = table[(table.Metric == 'RMSE') & (table.Model == 'ARGONet') & (table.State == st)]['ARGONet Period'].values[0]
    #     argo_rmse.append(tmp)
    # # END BLOCK
    #
    # # ARRAYS FOR CORRELATION (PANEL 2)
    # ar_corr = []
    # for st in states_corr_order:
    #     tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['ARGONet Period'].values[0]
    #     ar_corr.append(tmp)
    # argt_corr = []
    # for st in states_corr_order:
    #     tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'Net') & (table.State == st)]['ARGONet Period'].values[0]
    #     argt_corr.append(tmp)
    # argo_corr = []
    # for st in states_corr_order:
    #     tmp = table[(table.Metric == 'PEARSON') & (table.Model == 'ARGONet') & (table.State == st)]['ARGONet Period'].values[0]
    #     argo_corr.append(tmp)
    # # END BLOCK
    #
    # # ARRAYS FOR MAPE (PANEL 3)
    # ar_mape = []
    # for st in states_mape_order:
    #     tmp = table[(table.Metric == 'MAPE') & (table.Model == 'ARGO(gt,ath)') & (table.State == st)]['ARGONet Period'].values[0]
    #     ar_mape.append(tmp)
    # argt_mape = []
    # for st in states_mape_order:
    #     tmp = table[(table.Metric == 'MAPE') & (table.Model == 'Net') & (table.State == st)]['ARGONet Period'].values[0]
    #     argt_mape.append(tmp)
    # argo_mape = []
    # for st in states_mape_order:
    #     tmp = table[(table.Metric == 'MAPE') & (table.Model == 'ARGONet') & (table.State == st)]['ARGONet Period'].values[0]
    #     argo_mape.append(tmp)
    # # END BLOCK
    #
    #
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #
    # ind = np.arange(len(states))
    # width = 0.25
    #
    # p1 = ax1.bar(ind - width, ar_rmse, width, color='#ff715e')
    # p2 = ax1.bar(ind, argt_rmse, width, color='#9baaff')
    # p3 = ax1.bar(ind + width, argo_rmse, width, color='#519652')
    #
    # ax1.legend((p1[0], p2[0], p3[0]), ('ARGO(gt,ath)','Net','ARGONet'), frameon=False)
    # ax1.set_xticks(ind)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # # ax1.spines['bottom'].set_visible(False)
    # ax1.set_xticklabels(states_rmse_order)
    # ax1.set_ylabel('RMSE')
    # ax1.set_axisbelow(True)
    # ax1.grid(color='k', axis='y', linestyle=':', linewidth=1, alpha=0.25)
    # # ax1.set_ylim([0,2.7])
    # ax1.set_ylim([.05, 3])
    # ax1.set_xlim([-1, 37])
    # ax1.tick_params(axis='y', length=8, width=1.5)
    # ax1.tick_params(axis='x', length=4, width=1.5)
    # # ax1.set_yticks([0,0.9,1.8,2.7])
    # ax1.set_yscale('log')
    # ax1.locator_params(axis='y', tight=True)
    #
    # p4 = ax2.bar(ind - width, ar_corr, width, color='#ff715e')
    # p5 = ax2.bar(ind, argt_corr, width, color='#9baaff')
    # p6 = ax2.bar(ind + width, argo_corr, width, color='#519652')
    #
    # ax2.set_xticks(ind)
    # ax2.set_xticklabels(states_corr_order)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.set_ylabel('Correlation')
    # ax2.set_axisbelow(True)
    # ax2.grid(color='k', axis='y', linestyle=':', linewidth=1, alpha=0.25)
    # ax2.set_ylim([0,1])
    # ax2.set_xlim([-1,37])
    # ax2.tick_params(axis='y', length=8, width=1.5)
    # ax2.tick_params(axis='x', length=4, width=1.5)
    # ax2.locator_params(axis='y', tight=True, nbins=4)
    #
    # p7 = ax3.bar(ind - width, ar_mape, width, color='#ff715e')
    # p8 = ax3.bar(ind, argt_mape, width, color='#9baaff')
    # p9 = ax3.bar(ind + width, argo_mape, width, color='#519652')
    #
    # ax3.set_xticks(ind)
    # ax3.set_xticklabels(states_mape_order)
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.set_ylabel('MAPE')
    # ax3.set_axisbelow(True)
    # ax3.grid(color='k', axis='y', linestyle=':', linewidth=1, alpha=0.25)
    # ax3.set_xlim([-1,37])
    # # ax3.set_ylim([0,2.7])
    # ax3.set_ylim([0.08,3])
    #
    # ax3.tick_params(axis='y', length=8, width=1.5)
    # ax3.tick_params(axis='x', length=4, width=1.5)
    # # ax3.set_yticks([0,0.9,1.8,2.7])
    # ax3.locator_params(axis='y', tight=True)
    #
    # ax3.set_yscale('log')
    # # ax3.set_yticks([0.1, 0.5, 1])
    #
    #
    # for axis in ['bottom','left']:
    #     ax1.spines[axis].set_linewidth(1.5)
    #     ax2.spines[axis].set_linewidth(1.5)
    #     ax3.spines[axis].set_linewidth(1.5)
    #
    # f.set_size_inches(18, 9)
    # # f.savefig('bar3.png', dpi=300)
    # plt.show()

if __name__ == '__main__':
    main()
