import pandas as pd

df = pd.read_csv('/Users/batuhanhidiroglu/Desktop/internship/task1/task1/data/power_usage_2016_to_2020.csv')
df_weather = pd.read_csv('/Users/batuhanhidiroglu/Desktop/internship/task1/task1/data/weather_2016_2020_daily.csv')


n = df.shape[0]

p1 = pd.Series(range(n), index=pd.period_range('2016-06-01 00:00:00', freq='1h', periods=n))

df['StartDate'] = p1.to_frame().index

df.to_csv('updated_power_usage_2016_to_2020.csv', index=False)


m = df_weather.shape[0]

p2 = pd.Series(range(m), index=pd.period_range('2016-06-01', freq='1D', periods=m))

df_weather['Date'] = p2.to_frame().index

df_weather.to_csv('updated_weather_2016_2020_daily.csv', index=False)

