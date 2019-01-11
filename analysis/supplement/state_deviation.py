import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib import config


VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']


vals = []
for i, state in enumerate(VALID_STATES):

    state_dat = pd.read_csv(config.STATES_DIR + '/{0}/top_ens_preds.csv'.format(state), parse_dates=[0])
    ili = state_dat['ILI'].values

    ili = ili[ili != 0]

    dev = np.std(ili) / np.mean(ili)
    vals.append(np.round(dev, 2))

print zip(VALID_STATES, vals)
