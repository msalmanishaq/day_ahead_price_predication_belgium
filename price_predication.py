import streamlit as st
import pandas as pd
import numpy as np
import holidays
from darts import TimeSeries
from darts.models import NaiveSeasonal, RegressionModel
from darts.metrics import mape, rmse, mae
import plotly.express as px
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('Combined_data_fetch.csv')

# Convert 'datetime' column to a datetime data type
df['datetime'] = pd.to_datetime(df['datetime'])
# Set 'datetime' column as the index
df.set_index('datetime', inplace=True)
# Resample at 1-hour intervals
df = df.resample('1H').mean()
# Reset the index if you want the 'datetime' column to be a regular column again
df.reset_index(inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.set_index('datetime')

# Read another CSV file
df1 = pd.read_csv('day_ahead_prices.csv')
df1 = df1.rename(columns={'Unnamed: 0': 'datetime', '0': 'day_ahead_prices'})

df1['datetime'] = pd.to_datetime(df1['datetime'], utc=True)
df1 = df1.set_index('datetime')

# Merge the two DataFrames
merged_df = df.merge(df1, how='left', on='datetime')
merged_df['day_ahead_prices'].fillna(0, inplace=True)

mean_value = df['wind_federal_dayaheadforecast'].mean()
merged_df['wind_federal_dayaheadforecast'].fillna(mean_value, inplace=True)

mean_value = df['day_ahead_load'].mean()
merged_df['day_ahead_load'].fillna(mean_value, inplace=True)

# Set up holidays
merged_df = merged_df.reset_index()
merged_df['Day_of_week'] = merged_df['datetime'].dt.dayofweek
merged_df['Hour_of_day'] = merged_df['datetime'].dt.hour
merged_df = merged_df.set_index('datetime')
belgian_holidays = holidays.CountryHoliday('BE')
merged_df['Holidays'] = [int(date in belgian_holidays) for date in merged_df.index.date]

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Function to evaluate the model
def eval_model(model, past_covariates=None, future_covariates=None, to_retrain=True):
    backtest = model.historical_forecasts(series=price, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=pd.to_datetime('2023-09-01 00:00:00'), 
                                          retrain=to_retrain,
                                          verbose=False, 
                                          forecast_horizon=24)
    
    price[-len(backtest)-336:].plot()
    backtest.plot(label='Backtest')
    
    # Calculate Mean Absolute Error (MAE)
    error = price[-len(backtest):].pd_series() - backtest.pd_series()
    mae = error.abs().mean()
    
    print('Backtest MAE = {}'.format(mae))
    plt.show()

    error.rolling(49, center=True).mean().plot()
    plt.show()

# Read and preprocess your data as you did in your initial code

## Function to evaluate the model
def eval_model(model, past_covariates=None, future_covariates=None, to_retrain=True):
    backtest = model.historical_forecasts(series=price, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=pd.to_datetime('2023-09-01 00:00:00'), 
                                          retrain=to_retrain,
                                          verbose=False, 
                                          forecast_horizon=24)
    
    price[-len(backtest)-336:].plot()
    backtest.plot(label='Backtest')
    
    # Calculate Mean Absolute Error (MAE)
    error = price[-len(backtest):].pd_series() - backtest.pd_series()
    mae = error.abs().mean()
    
    print('Backtest MAE = {}'.format(mae))
    
    # Plot the rolling mean of the error
    error.rolling(49, center=True).mean().plot()
    plt.show()
    
    return mae

# Read and preprocess your data as you did in your initial code

# Load your time series data
price = TimeSeries.from_dataframe(merged_df, value_cols=['day_ahead_prices'], freq=None)
price_train, price_test = price.split_after(pd.to_datetime('2023-09-01 00:00:00'))
# Train a model (e.g., Naive Seasonal)
naive_model = NaiveSeasonal(K=168)
naive_model.fit(price_train)
prediction = naive_model.predict(len(price_test))

# Train a regression model
future = TimeSeries.from_dataframe(merged_df, value_cols=['wind_federal_dayaheadforecast', 'wind_Flanders_dayaheadforecast', 'wind_Wallonia_dayaheadforecast', 'day_ahead_load', 'Solar_day_ahead_forecasted', 'Day_of_week', 'Hour_of_day', 'Holidays'])

regr_model_cov = RegressionModel(lags=list(range(-24, 0)), lags_future_covariates=list(range(24)))
regr_model_cov.fit(price_train, future_covariates=future)

# Set up Streamlit
st.title("Machine Learning Model Deployment")
st.subheader("Predicted Values for the Test Set")

# Display actual and predicted values using Plotly
st.write("Actual Prices:")
fig_actual = px.line(price_test.pd_dataframe(), x=price_test.pd_dataframe().index, y='day_ahead_prices', title="Actual Prices")
st.plotly_chart(fig_actual, use_container_width=True)

st.write("Predicted Prices:")
fig_predicted = px.line(prediction.pd_dataframe(), x=prediction.pd_dataframe().index, y='day_ahead_prices', title="Predicted Prices")
st.plotly_chart(fig_predicted, use_container_width=True)

# Checkbox to show/hide the dataset and its summary statistics
show_dataset = st.checkbox("Show Dataset", value=False)

if show_dataset:
    st.subheader("Merged Dataset")
    st.write(merged_df)

    st.subheader("Summary Statistics for Merged Dataset")
    st.write(merged_df.describe())

# Evaluate the regression model
mae_value = eval_model(regr_model_cov, future_covariates=future, to_retrain=False)

# Display MAE value
st.subheader("Model Evaluation")
st.write(f"MAE: {mae_value:.2f}")
