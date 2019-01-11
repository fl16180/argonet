import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib import config

# plt.style.use('seaborn-poster')

# class config():
#     DATA_DIR = 'C:/Users/fredl/Documents/repos/state-flu/data/'

VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']


ili = pd.read_csv(config.DATA_DIR + '/analysis/ILINet.csv', header=1)

ili2 = ili.loc[(ili.YEAR >= 2014) & (ili.YEAR <=2017)]
ili2 = ili2.loc[ ~((ili2.YEAR == 2014) & (ili2.WEEK < 40)) ]
ili2 = ili2.loc[ ~((ili2.YEAR == 2017) & (ili2.WEEK > 20)) ]

ili2.tail()


abbrevs = pd.read_csv(config.DATA_DIR + '/GFT/state_abbrev.csv')
ili2['State'] = ili2.REGION
ili2 = ili2.merge(abbrevs)
ili2.rename(columns={'NUM. OF PROVIDERS':'providers'}, inplace=True)
ili2.head()

ili_clean = ili2.loc[:, ['ST','YEAR','WEEK','providers']]
ili_clean['YRWK'] = ili_clean['YEAR'].map(str) + ili_clean['WEEK'].map(str)

final = ili_clean.pivot(index='YRWK', columns='ST', values='providers').reset_index().drop('YRWK',axis=1)
final2 = final.loc[:, final.columns.isin(VALID_STATES)]
final2 = final2.astype(int)
final2.head()

sns.boxplot(data=final2)
plt.xlabel('State')
plt.ylabel('Providers')



final2.mean()
