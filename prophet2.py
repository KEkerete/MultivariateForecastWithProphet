# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 23:35:11 2023

@author: keker
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from prophet import Prophet

import matplotlib.pyplot as plt


# Load your data into a Pandas DataFrame
dataTemp = pd.read_csv('E:\CEDA_Data\\MetNFadeDataChilbolton.csv')

# extract needed columns from fade file
df = dataTemp[['ds', 'y','ChilboltonPressure', 'ChilboltonWindSpeed', 'ChilboltonWindDirection','ChilboltonDropCount']] #,'ChilboltonTemperature','ChilboltonDewPoint']]
# df.dropna(subset=['ChilboltonTemperature'])

# Specify the time series feature you want to forecast
y = df['y']

# Specify the features that will be used as regressors
X = df[['ChilboltonPressure', 'ChilboltonWindSpeed', 'ChilboltonWindDirection', 'ChilboltonDropCount']] #,'ChilboltonTemperature','ChilboltonDewPoint']]


# Fit the Prophet model using the time series and regressor features

# initialize the Prophet model with the correct frequency of observations
# model = Prophet()
model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1)
model.add_seasonality(name='10 seconds', period=10, fourier_order=5)

# Add the independent variables to the model
for col in ['ChilboltonPressure','ChilboltonWindSpeed','ChilboltonWindDirection','ChilboltonDropCount']:  #,'ChilboltonTemperature','ChilboltonDewPoint']:
    model.add_regressor(col)

# fit the model to the data
model.fit(df)

# generate the forecast for the next 6 minutes
future = model.make_future_dataframe(periods=360, freq='10S')

# Specify the values for the regressor features in the future values
future['ChilboltonPressure'] = df['ChilboltonPressure']
future['ChilboltonWindSpeed']= df['ChilboltonWindSpeed']
future['ChilboltonWindDirection'] = df['ChilboltonWindDirection']
future['ChilboltonDropCount'] = df['ChilboltonDropCount']   #,'ChilboltonTemperature','ChilboltonDewPoint']:

future['y'] = df['y']   #,'ChilboltonTemperature','ChilboltonDewPoint']:
y1 = future['y']
    
# =============================================================================
## Using a machine learning model to predict the values:
# from sklearn.ensemble import RandomForestRegressor
# 
# # Train a machine learning model on the regressor data
# regressor = RandomForestRegressor()
# regressor.fit(df[['ds']], df['regressor1'])
# 
# # Use the trained model to predict the values for the future values
# future['regressor1'] = regressor.predict(future[['ds']])
# =============================================================================


# Make the forecast
forecast = model.predict(future)

# Extract the forecast for the time series feature
y_forecast = forecast['yhat']
x_axis = forecast['ds']

############ plot forecast
# plot the forecast for the target variable
# model.plot(y_forecast)
plt.plot(x_axis, y1)
plt.plot(x_axis, y_forecast)

# add a title to the plot
plt.title('Forecast for Fade')

# show the plot
plt.show()

stop
########## evaluate performance
# load the actual values into a pandas DataFrame
actual = df   # pd.read_csv("actual.csv")

# set the date column as the index and convert it to the correct datetime format
actual['ds'] = pd.to_datetime(actual['ds'], format='%d-%m-%Y %H:%M')
actual.set_index('ds', inplace=True)

# select the target variable from the actual values
actual_values = actual['y']

# select the corresponding forecasted values from the forecast
forecasted_values = forecast[['ds', 'yhat']]
forecasted_values = forecasted_values.set_index('ds')
forecasted_values = forecasted_values['yhat']

# calculate the MAE
mae = mean_absolute_error(actual_values, forecasted_values)

# calculate the MSE
mse = mean_squared_error(actual_values, forecasted_values)

# calculate the RMSE
rmse = np.sqrt(mse)

# print the results
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

######### 
# use cross-validation to evaluate the performance of the Prophet model using the 'sklearn' library

# load the actual values into a pandas DataFrame
actual = pd.read_csv("actual.csv")

# set the date column as the index and convert it to the correct datetime format
actual['ds'] = pd.to_datetime(actual['date'], format='%d-%m-%Y %H:%M')
actual.set_index('ds', inplace=True)

# select the target variable from the actual values
actual_values = actual['target_variable']

# set up a time series cross-validation object
tscv = TimeSeriesSplit(n_splits=5)

# initialize the MAE, MSE, and RMSE values
mae = 0
mse = 0
rmse = 0

# loop over the folds in the cross-validation
for train_index, test_index in tscv.split(actual_values):
    
    # select the training and testing data for the target variable
    train_target = actual_values.iloc[train_index]
    test_target = actual_values.iloc[test_index]
    
    # fit the model on the training data
    model = Prophet()
    model.fit(train_target)
    
    # make predictions on the testing data
    future = model.make_future_dataframe(periods=len(test_target), freq='10S')
    forecast = model.predict(future)
    forecasted_values = forecast.iloc[-len(test_target):]['yhat']
    
    # update the MAE, MSE, and RMSE values with the results from this fold
    mae += mean_absolute_error(test_target, forecasted_values)
    mse += mean_squared_error(test_target, forecasted_values)
    rmse += np.sqrt(mean_squared_error(test_target, forecasted_values))

# average the MAE, MSE, and RMSE values over all folds
mae /= 5
mse /= 5
rmse /= 5

# print the results
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)











# set the date column as the index and convert it to the correct datetime format
# df['ds'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
# df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M')
# df.set_index('ds', inplace=True)


# Define the columns for the independent and dependent variables
# df.columns = ['ds', 'y', 'x1', 'x2', ...]  # ,	'ChilboltonAtt',
df.columns = ['ds', 'y','ChilboltonPressure', 'ChilboltonWindSpeed', 'ChilboltonWindDirection', 'ChilboltonDropCount'] #,'ChilboltonTemperature','ChilboltonDewPoint']


# Create a Prophet model object
#m = Prophet()
# initialize the Prophet model with the correct frequency of observations
model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1)
model.add_seasonality(name='10 seconds', period=10, fourier_order=5)


# Add the independent variables to the model
for col in ['ChilboltonPressure','ChilboltonWindSpeed','ChilboltonWindDirection','ChilboltonDropCount']:  #,'ChilboltonTemperature','ChilboltonDewPoint']:
    model.add_regressor(col)


# fit the model to the data
model.fit(df)

# generate the forecast for the next 6 minutes
future = model.make_future_dataframe(periods=360, freq='10S')
# prophet_forecast_step['ex'] = make_future_dataframe(periods=0)

