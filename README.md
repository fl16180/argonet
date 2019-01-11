## Overview
This project contains the code supporting the findings of "Lu et al. (2019). *Improved state-level influenza nowcasting in the United States leveraging Internet-based data and network approaches.*" published at https://www.nature.com/articles/s41467-018-08082-0. The paper reports a methodology for accurate near-real time influenza prediction within each state. The eponymous model of this repository, ARGONet, is an ensemble of two component models: ARGO, which uses web-based data correlated with flu activity, and Net, which makes predictions based on network synchrony between states.

## Setup

Clone or download the repository. The code was written in Python 2.7 with Anaconda package distribution. Package requirements for modeling are numpy, pandas, scikit-learn, scipy. Visualization uses matplotlib and seaborn except for geographical heatmaps which use ggplot2 in R 3.4.  

## Data

The data directory contains the original data files compiled from Google Trends, ILINet, and Google Flu Trends. The electronic health records data in its raw form is not present for privacy reasons. However, the relevant rates computed from the raw data are available and saved in the preprocessing step. For raw data files pre-compiling, refer to the Harvard dataverse archive linked in our paper.

## Preprocessing
The relevant time series from the data were preprocessed and concatenated into a separate csv for each state, located in the [results](./results/) directory. The main csv is labeled in the form "XX_merged_data.csv", and the scaled Google Flu Trends time series is separately saved as "GFT_scaled.csv". The scripts used for this purpose are stored in [preprocessing](./lib/preprocessing/).


## Modeling
The main results of the paper use the above preprocessed data files as input. The code for reproducing the modeling steps are in the [exec](./exec) directory and should be used in the following order:

1. [full_data_models.py](./exec/full_data_models.py) runs a set of calls to the ARGO library in [forecastlib](./lib/forecastlib). These run the AR52 benchmark, standard ARGO, and modifications to the ARGO algorithm as documented in the paper. The results are stored in each state's own directory in [results](./results), named "ath_preds.csv" and "ath_table.csv".

2. [compile_best_experiments.py](./exec/compile_best_experiments.py) selects the best model from the previous step and stores its predictions in the results directory, named "top_argo_preds.csv" and "top_argo_table.csv"

3. [net_model.py](./exec/net_model.py) runs the Net model.

4. [ensemble_model.py](./exec/ensemble_model.py) generates the ensemble predictions from the ARGO and Net prediction files.

5. [append_models.py](./exec/append_models.py) maps the Net and ensemble predictions from steps 3 and 4 back to the individual state prediction tables.


## Visualization

Refer to the [analysis](./analysis) directory. Many of the plots and figures used in our paper rely on a compiled table of computed metrics of all the models. To generate this table, run [merge_ensemble_tables.py](./analysis/merge_ensemble_tables.py). This script compiles multiple tables, but the most salient one is 'final_table_merged.xlsx'.

The most relevant visualization is [final_state_plot.py](./analysis/final_state_plot.py) which generates the time series plot (Fig. 4). This is the cleanest way to check the output of the models by eye. Our other figures were generated piecemeal from the various scripts in the directory, and I have not gone through and re-organized them.
