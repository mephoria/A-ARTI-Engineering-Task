import pandas as pd

df = pd.read_csv('data/prev_versions_data/absolute_data_v1.csv')

df['StartDate'] = pd.to_datetime(df['StartDate'])
df['Hour_of_Day'] = df['StartDate'].dt.hour

df.to_csv('data/data_v2.csv', index=False)