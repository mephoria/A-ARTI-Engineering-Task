import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.api.types import CategoricalDtype

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv('data/data_v3.csv', index_col=[0], parse_dates=[0])

print("Initial columns:", df.columns)
columns_to_keep = {'StartDate', 'Value (kWh)', 'day_of_week_x', 'Hour_of_Day', 'notes'}
all_names = {'Value (kWh)', 'day_of_week_x', 'Temp_max', 'Temp_avg', 'Temp_min',
             'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max', 'Hum_avg', 'Hum_min',
             'Wind_max', 'Wind_avg', 'Wind_min', 'Press_max', 'Press_avg',
             'Press_min', 'Precipit', 'HDD', 'CDD', 'Hour_of_Day', 'notes'}
df.drop(columns=list(all_names - columns_to_keep), inplace=True)
print("Columns after dropping:", df.columns)

color_pal = sns.color_palette()
df.plot(style='.', figsize=(10, 5), ms=1, color=color_pal[0], title='Power Use')
plt.savefig('all_time_use.png')

cat_type = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)

def create_features(df, label=None):
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date_offset'] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300
    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'weekday', 'season']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(df, label='Value (kWh)')
features_and_target = pd.concat([X, y], axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=features_and_target.dropna(), x='weekday', y='Value (kWh)', hue='season', ax=ax, linewidth=1)
ax.set_title('Power Use by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Power (kWh)')
ax.legend(bbox_to_anchor=(1, 1))
plt.savefig('weekly_data.png')

# Train-test split
split_date = '2020-05-07'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()

df_test.rename(columns={'Value (kWh)': 'TEST SET'}, inplace=True)
df_train.rename(columns={'Value (kWh)': 'TRAINING SET'}, inplace=True)

df_combined = df_test[['TEST SET']].join(df_train[['TRAINING SET']], how='outer')
df_combined.plot(figsize=(15, 5), title='Power Use', style='.')
plt.savefig('train-test_split.png')

df_train_prophet = df_train.reset_index().rename(columns={'StartDate': 'ds', 'TRAINING SET': 'y'})
df_test_prophet = df_test.reset_index().rename(columns={'StartDate': 'ds', 'TEST SET': 'y'})

model = Prophet()
model.fit(df_train_prophet)

df_test_forecast = model.predict(df_test_prophet[['ds']])

print("df_test_forecast columns:", df_test_forecast.columns)

fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(df_test_forecast, ax=ax)
ax.set_title('Forecast')
plt.savefig('test_forecast.png')

f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(df_test.index, df_test['TEST SET'], color='r')
fig = model.plot(df_test_forecast, ax=ax)
ax.set_title('Comparison w/ real data')
plt.savefig('comparison_prophet.png')

df_test_forecast.set_index('ds', inplace=True)
df_test_combined = df_test.join(df_test_forecast[['yhat']], how='left')

score = np.sqrt(mean_squared_error(df_test['TEST SET'], df_test_combined['yhat']))
print(f'RMSE Score on Test set: {score:0.2f}')
