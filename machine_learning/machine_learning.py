import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

df = pd.read_csv('data/data_v3.csv', index_col=[0], parse_dates=[0])
df.index = pd.to_datetime(df.index)
print("Initial columns:", df.columns)

columns_to_keep = {'StartDate', 'Value (kWh)', 'day_of_week_x', 'Hour_of_Day', 'notes'}
all_names = {'Value (kWh)', 'day_of_week_x', 'Temp_max', 'Temp_avg', 'Temp_min',
             'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max', 'Hum_avg', 'Hum_min',
             'Wind_max', 'Wind_avg', 'Wind_min', 'Press_max', 'Press_avg',
             'Press_min', 'Precipit', 'HDD', 'CDD', 'Hour_of_Day', 'notes'}

df.drop(columns=list(all_names - columns_to_keep), inplace=True)
print("Columns after dropping:", df.columns)

color_pal = sns.color_palette()
df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='Energy Use')
plt.savefig('initial_look.png')

# Train-test split
split_date = pd.Timestamp('2020-05-07')

train = df.loc[df.index < split_date]
test = df.loc[df.index >= split_date]

fig, ax = plt.subplots(figsize=(15, 5))
train['Value (kWh)'].plot(ax=ax, label='Training Set')
test['Value (kWh)'].plot(ax=ax, label='Test Set')
ax.set_title('Train/Test Split')
ax.legend()
plt.savefig('test-train_split.png')

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'Value (kWh)'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['Value (kWh)']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.savefig('results_ml.png')


score = np.sqrt(mean_squared_error(test['Value (kWh)'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')