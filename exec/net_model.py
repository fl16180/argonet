import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from lib import config
from compile_prediction_dataframe import compile
from lib.forecastlib.processing_functions import create_ar_stack


def main():
    INPUT = 'top_argo_preds'
    OUTPUT = 'all_states_argo'
    compile(INPUT, OUTPUT)

    argo = pd.read_csv(config.STATES_DIR + '/_overview/{0}.csv'.format(OUTPUT))
    argo.head()

    states = argo.state.unique()
    states
