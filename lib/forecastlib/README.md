forecastlib is a Python 2.7 library that provides a simple and streamlined machine learning flow for 
the ARGO disease prediction model. The package implements an object-oriented model for time series 
regression using multiple data sources in real-time.

ARGO models a time series target using a combination of endogenous historical data (autoregressive model)
and exogenous time series variables from various sources (e.g. Google Trends).
The time series target can represent ILI (influenza-like illness) rates, flu visit counts, or any other
single-column variable, and is read from a csv file containing a date column and the variable column. 
The exogenous variables are read from csv files containing a date column and one or multiple variable columns.

The following exogenous time series data sources are supported: athenahealth, Google Trends, Flu Near You.

The following demo shows the flow for loading data to generate ILI predictions for the state of Massachusetts.

>>>	import forecastlib as forecast
>>>	state = forecast.Geo_model(geo_level='State', geo_id='MA', resolution='week', pred_start_date='2014-01-05', 
				use_data_from='2010-10-03')

>>>	state.data_load(target_file='insert-location', gt_file='insert-location',
				ath_file='insert-location', fny_file='insert-location')

				
To start a prediction model, an initialization call is used, which returns a predictor object. Specific processing
such as transforms can be applied to this object.

>>>	pred = state.init_predictor('athenahealth')
>>>	pred.data_process(transform_target=False, transform_ath=False, AR_terms=52)
	

To run a prediction, specify the data sources to be used, the method, and the time horizon 
(0 = nowcast, 1 = 1 wk forecast, etc).

>>>	pred.predict(['ath'], method='ARGO', horizon=0, train_len='default', in_sample=False)


Running no prediction at all is also accepted for cases where an input variable	is to be used directly as 
the prediction.

>>>	no_pred = state.init_predictor('FNY')
>>>	no_pred.predict(['fny'], method='None')


The following is an example of predicting using multiple data sources

>>>	combined_pred = state.init_predictor('Forecast')
>>>	combined_pred.data_process(AR_terms=52)
>>>	combined_pred.predict(['ath', 'fny', 'gt'], method='Ridge', horizon=0, train_len='default', in_sample=False)


To save the predictions to a csv file, execute

>>>	state.save_predictions(fname='insert-location')


For detailed descriptions of parameters, see the documentation for models.py
