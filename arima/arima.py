import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('data/data_v3.csv', index_col=[0], parse_dates=[0])
df.index = pd.to_datetime(df.index)
# print("Initial columns:", df.columns)

columns_to_keep = {'StartDate', 'Value (kWh)'}
all_names = {'Value (kWh)', 'day_of_week_x', 'Temp_max', 'Temp_avg', 'Temp_min',
             'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max', 'Hum_avg', 'Hum_min',
             'Wind_max', 'Wind_avg', 'Wind_min', 'Press_max', 'Press_avg',
             'Press_min', 'Precipit', 'HDD', 'CDD', 'Hour_of_Day', 'notes'}

df.drop(columns=list(all_names - columns_to_keep), inplace=True)
# print("Columns after dropping:", df.columns)

# Train-test split
split_date = pd.Timestamp('2020-05-07')

df_train = df.loc[df.index < split_date]
df_test = df.loc[df.index >= split_date]


# Find out whether time series is stationary or not: Is stationary
# acf_original = plot_acf(df_train)
# pacf_original = plot_pacf(df_train)
# adf_test = adfuller(df_train)
# print(f"p-value: {adf_test[1]}")
 
model = ARIMA(df_train, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Check residuals and density# 
# residuals = model_fit.resid[1:]
# fig, ax = plt.subplots(1, 2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(title="Density", kind="kde", ax=ax[1])


#forecast_test = model_fit.forecast(len(df_test))
#df["forecast_manual"] = [None]*len(df_train) + list(forecast_test)
#df[['Value (kWh)', 'forecast_manual']].plot(figsize=(10, 6))
#plt.show()

# Forecast
forecast_test = model_fit.forecast(steps=len(df_test))
df["forecast_manual"] = np.nan
df.loc[df_test.index, "forecast_manual"] = forecast_test.values

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Value (kWh)'], label='Actual')
plt.plot(df.index, df['forecast_manual'], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value (kWh)')
plt.title('Actual and Forecasted Values')
plt.legend()
plt.show()