## Overview
This project contains the code supporting the findings of "Improved state-level influenza nowcasting in the United States
leveraging Internet-based data and network approaches" by *Lu et al (2018)*. The paper reports a methodology for accurate near-real time influenza prediction within each state. The eponymous model of this repository, ARGONet, is an ensemble of two component models: ARGO, which uses web-based data correlated with flu activity, and Net, which makes predictions based on network synchrony between states.

## Setup

Download the repository. The code was written in Python 2.7 with Anaconda package distribution. Package requirements for modeling are numpy, pandas, scikit-learn, scipy. Visualization uses matplotlib and seaborn except for geographical heatmaps which use ggplot2 in R 3.4.  

## Data

The data directory contains the original data files compiled from Google Trends, ILINet, and Google Flu Trends. The electronic health records data in its raw form is not present for privacy reasons. However, the relevant rates computed from the raw data are available and saved in the preprocessing step.

## Preprocessing
The relevant time series from the data were preprocessed and concatenated into a separate csv for each state, located in the [results](./results/) directory. The main csv is labeled in the form "XX_merged_data.csv", and the scaled Google Flu Trends time series is separately saved as "GFT_scaled.csv". The scripts used for this purpose are stored in [preprocessing](./lib/preprocessing/).


## Modeling
The main results of the paper use the above preprocessed data files as input. The code for reproducing the modeling steps are in the [exec](./exec) directory and should be used in the following order:

1. [full_data_models.py](./exec/full_data_models.py) runs a set of calls to the ARGO library in [forecastlib](./lib/forecastlib). These run the AR52 benchmark, standard ARGO, and modifications to the ARGO algorithm as documented in the paper. The results are stored in each state's own directory in [results](./results), named "ath_preds.csv" and "ath_table.csv".

2. [compile_best_experiments.py](./exec/compile_best_experiments.py) selects the best model from the previous step and stores its predictions in the results directory, named "top_argo_preds.csv" and "top_argo_table.csv"

3. [net_model.py](./exec/net_model.py) runs the Net model.

4. [ensemble_model.py](./exec/ensemble_model.py) generates the ensemble predictions from the ARGO and Net prediction files.

## Visualization
Once predictions are generated, results can be visualized using scripts in [final_analysis](./final_analysis). I will update this section with details once the paper is published. 
