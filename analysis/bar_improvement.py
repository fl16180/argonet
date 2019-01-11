'''
'''

from __future__ import division
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

from lib import config


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def main():

    # read in table with pre-computed metrics over each season and state
    table = pd.read_csv(config.STATES_DIR + '/_overview/final_table_merged.csv')
    table2 = pd.read_csv(config.STATES_DIR + '/_overview/final_table_ens.csv')


    states = list(set(table.State.values))

    # limit to specific metrics
    table_rmse = table[table.Metric == 'RMSE']
    table_corr = table[table.Metric == 'PEARSON']
    table_mape = table[table.Metric == 'MAPE']

    table2_rmse = table2[table2.Metric == 'RMSE']
    table2_corr = table2[table2.Metric == 'PEARSON']
    table2_mape = table2[table2.Metric == 'MAPE']



    ####################################################
    ############         Figure 3           ############
    ####################################################

    # get list of differences between RMSEs
    rmse_diffs = [float(table_rmse[(table_rmse.Model == 'ARGO') & (table_rmse.State == st)]['ARGONet Period'].values[0]) \
            / float(table_rmse[(table_rmse.Model == 'Net') & (table_rmse.State == st)]['ARGONet Period'].values[0])
            for st in states]


    # reorder states with respect to the ordered diffs
    states_rmse_order = [states[i] for i in argsort(rmse_diffs)[::-1]]
    # states_corr_order = [states[i] for i in argsort(corr_diffs)]
    # states_mape_order = [states[i] for i in argsort(mape_diffs)[::-1]]

    ### GENERATE ARRAYS FOR BAR PLOTS ###

    # ARRAYS FOR RMSE (PANEL 1)
    argo_rmse = []
    for st in states_rmse_order:
        tmp = table[(table.Metric == 'RMSE') & (table.Model == 'ARGO') & (table.State == st)]['ARGONet Period'].values[0]
        argo_rmse.append(tmp)
    net_rmse = []
    for st in states_rmse_order:
        tmp = table[(table.Metric == 'RMSE') & (table.Model == 'Net') & (table.State == st)]['ARGONet Period'].values[0]
        net_rmse.append(tmp)
    ar_rmse = []
    for st in states_rmse_order:
        tmp = table[(table.Metric == 'RMSE') & (table.Model == 'AR52') & (table.State == st)]['ARGONet Period'].values[0]
        ar_rmse.append(tmp)
    # END BLOCK






    # get list of differences between RMSEs
    rmse_diffs2 = [table2_rmse[(table2_rmse.Model == 'ARGO') & (table2_rmse.State == st)]['ARGONet Period'].values[0] \
            / table2_rmse[(table2_rmse.Model == 'ARGONet') & (table2_rmse.State == st)]['ARGONet Period'].values[0]
            for st in states]



    # reorder states with respect to the ordered diffs
    states_rmse_order2 = [states[i] for i in argsort(rmse_diffs2)[::-1]]


    ### GENERATE ARRAYS FOR BAR PLOTS ###

    # ARRAYS FOR RMSE (PANEL 1)
    argo_rmse2 = []
    for st in states_rmse_order2:
        tmp = table2[(table2.Metric == 'RMSE') & (table2.Model == 'ARGO') & (table2.State == st)]['ARGONet Period'].values[0]
        argo_rmse2.append(tmp)
    argonet_rmse2 = []
    for st in states_rmse_order2:
        tmp = table2[(table2.Metric == 'RMSE') & (table2.Model == 'ARGONet') & (table2.State == st)]['ARGONet Period'].values[0]
        argonet_rmse2.append(tmp)
    ar_rmse2 = []
    for st in states_rmse_order2:
        tmp = table2[(table2.Metric == 'RMSE') & (table2.Model == 'AR52') & (table2.State == st)]['ARGONet Period'].values[0]
        ar_rmse2.append(tmp)
    # END BLOCK





    from math import sqrt

    def filt(x, y):
        mat = np.array(zip(x, y))
        mat = mat[~np.any(mat==0, axis=1) & ~np.any(mat==-.001, axis=1)]
        return mat[:, 0], mat[:, 1]

    # scoring metric functions
    def RMSE(predictions, targets):
        predictions, targets = filt(predictions, targets)
        return sqrt(((predictions - targets) ** 2).mean())



    # error_bars = np.zeros((2, len(states_rmse_order)))
    # for state_num, state in enumerate(states_rmse_order):
    #     preds = pd.read_csv(config.STATES_DIR + '/{0}/top_ens_preds.csv'.format(state))
    #
    #     target = preds['ILI'].values
    #     argo = preds['ARGO(gt,ath)'].values
    #     argonet = preds['ARGONet'].values
    #
    #
    #     methods = np.column_stack((target, argo, argonet))
    #
    #
    #     # geometric probability = 1 / mean length
    #     p = 1./52
    #
    #     samples = 1000
    #     n_models = methods.shape[1] - 1
    #
    #     # calculate observed relative efficiency
    #     eff_obs = np.zeros(n_models)
    #     for i in range(n_models):
    #     	eff_obs[i] = RMSE(methods[:, 0], methods[:, i + 1])
    #     eff_obs = eff_obs / eff_obs[-1]
    #
    #     # perform bootstrap
    #     scores = np.zeros((samples, n_models))
    #     for iteration in range(samples):
    #     	# construct bootstrap resample
    #     	new, n1, n2 = sbb.resample(methods, p)
    #
    #     	# calculate sample relative efficiencies
    #     	for model in range(n_models):
    #     		scores[iteration, model] = RMSE(new[:, 0], new[:, model + 1])
    #     	scores[iteration] = scores[iteration] / scores[iteration, -1]
    #
    #     # define the variable containing the deviations from the observed rel eff
    #     scores_residual = scores - eff_obs
    #
    #     # construct output array
    #     report_array = np.zeros((3,n_models))
    #     for comp in range(n_models):
    #     	tmp = scores_residual[:, comp]
    #
    #     	# 90% confidence interval by sorting the deviations and choosing the endpoints of the 95% region
    #     	tmp = np.sort(tmp)
    #     	report_array[0, comp] = eff_obs[comp]
    #     	report_array[1, comp] = eff_obs[comp] + tmp[int(round(samples * 0.05))]
    #     	report_array[2, comp] = eff_obs[comp] + tmp[int(round(samples * 0.95))]
    #
    #     print report_array.T
    #
    #     error_bars[0, state_num] = report_array[0, 0] - report_array[1, 0]
    #     error_bars[1, state_num] = report_array[2, 0] - report_array[0, 0]






    ind = np.arange(len(states))

    from matplotlib import cm

    c_range = np.linspace(.2, .8, len(ind))

    # c_range[np.abs(c_range - 0.5) < 0.05] -= 0.05 * np.sign(c_range[np.abs(c_range - 0.5) < 0.05])
    cmap = cm.RdBu_r(c_range)

    improvement = np.array([float(argo_rmse[i]) / float(net_rmse[i]) for i in ind])
    improvement2 = np.array([argo_rmse2[i] / argonet_rmse2[i] for i in ind])


    colors = cm.RdBu(4 * (improvement - 1) / max(improvement) + 0.5)
    colors2 = cm.RdBu(4 * (improvement2 - 1) / max(improvement2) + 0.5)

    # print colors
    f, (ax1, ax2) = plt.subplots(2, 1)

    p = ax1.scatter(ind, improvement - 1, s=50, color=colors, edgecolor='k', linewidth=0.5)
    ax1.set_ylabel('Improvement of Net', labelpad=14, fontsize=12)
    ax1.set_ylim([-0.38, 0.38])
    ax1.set_xlim([-1.5, 37])

    ax1.set_xticks(ind)
    ax1.set_xticklabels(states_rmse_order)

    p = ax2.scatter(ind, improvement2 - 1, s=50, color=colors2, edgecolor='k', linewidth=0.5, alpha=1)
    ax2.set_ylabel('Improvement of ARGONet', labelpad=14, fontsize=12)
    ax2.set_ylim([-0.38, 0.38])
    ax2.set_xlim([-1.5, 37])

    ax2.set_xticks(ind)
    ax2.set_xticklabels(states_rmse_order2)


    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_bounds(-.3, .3)
    ax1.spines['bottom'].set_bounds(-.5, 36.5)

    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1.5)

    ax1.tick_params(axis='y', length=4, width=1, rotation=0, labelsize=12)
    ax1.tick_params(axis='x', length=0, width=1, rotation=90, labelsize=9)
    ax1.set_yticks([-.3, 0, .3])
    ax1.set_yticklabels(['0.7', '1', '1.3'])
    # ax1.locator_params(axis='y', tight=True, nbins=4)

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.25)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_bounds(-.3, .3)
    ax2.spines['bottom'].set_bounds(-.5, 36.5)

    ax2.tick_params(axis='y', length=4, width=1, rotation=0, labelsize=12)
    ax2.tick_params(axis='x', length=0, width=1, rotation=90, labelsize=9)
    ax2.set_yticks([-.3, 0, .3])
    ax2.set_yticklabels(['0.7', '1', '1.3'])
    # ax2.locator_params(axis='y', tight=True, nbins=4)

    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['left'].set_linewidth(1.5)

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.25)

    # ax1.set_title("Net")
    # ax2.set_title("ARGONet")


    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.savefig(config.STATES_DIR + '/_overview/improvements.png', format='png', dpi=300)


    plt.show()











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
