import pandas as pd


df = pd.read_csv('/Users/batuhanhidiroglu/Desktop/internship/task1/task1/data/final_data.csv')
df['StartDate'] = pd.to_datetime(df['StartDate'])

# These columns are redundant as other columns convey this information as well
df.drop(columns=['Day', 'notes', 'day_of_week_y'], inplace=True)

weather_columns = ['Temp_max', 'Temp_avg', 'Temp_min', 'Dew_max', 'Dew_avg', 'Dew_min', 
                   'Hum_max', 'Hum_avg', 'Hum_min', 'Wind_max', 'Wind_avg', 'Wind_min', 
                   'Press_max', 'Press_avg', 'Press_min', 'Precipit']
df[weather_columns] = df[weather_columns].ffill()


# Fix 'day_of_week_x'
df['day_of_week_x'] = df['StartDate'].dt.dayofweek


df.to_csv('final_finaldata.csv', index=False)
