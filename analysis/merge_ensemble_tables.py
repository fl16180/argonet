import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from lib import config
from lib.analysis import master_table, master_table_net, master_table_ensemble

STATES_DIR = config.STATES_DIR

master_table.main()
master_table_net.main()
master_table_ensemble.main()

table = pd.read_csv(STATES_DIR + '/_overview/final_table_net.csv')
ens_table = pd.read_csv(STATES_DIR + '/_overview/final_table_ens.csv')

table.head(6)
ens_table.head(6)

table1 = table.set_index(['Metric','State','Model'])
ens_table1 = ens_table.set_index(['Metric','State','Model'])
table1.head()
ens_table1.head()
# final = table1.merge(ens_table1, how='outer')

final = pd.merge(table1[['2012-13','2013-14','2014-15','Whole Period','GFT Period']], ens_table1, how='outer', left_index=True, right_index=True)
final.head(10)

final = final.reindex(['RMSE','PEARSON','MAPE'], level=0)
final = final.reindex(['GFT','AR52','ARGO','Net','ARGONet'], level=2)
final = final.reset_index()
final.head(10)

final['2014-15_x'].fillna(final['2014-15_y'], inplace=True)
del final['2014-15_y']
final.rename(columns={'2014-15_x':'2014-15'}, inplace=True)

final.loc[final['Model'] == 'GFT', 'Whole Period'] = np.nan
cols = ['Metric','State','Model','2012-13','2013-14','2014-15','2015-16','2016-17','Whole Period','GFT Period','ARGONet Period']
final = final[cols]

final = final[final['Model'] != 'ARGO(gt)']
final.loc[final['Model'] == 'ARGO', 'Model'] = 'ARGO'

final = final.set_index(['Metric','State','Model'])

final[final == '--'] = np.nan
final['2016-17'] = final['2016-17'].astype(np.float)
print final.head(20)



final.to_csv(STATES_DIR + '/_overview/final_table_merged.csv', na_rep='--')
final.to_excel(STATES_DIR + '/_overview/final_table_merged.xlsx', na_rep='--')
