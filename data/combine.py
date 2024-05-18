import pandas as pd

df_power = pd.read_csv("/Users/batuhanhidiroglu/Desktop/internship/task1/task1/data/power_data.csv")
df_weather = pd.read_csv("/Users/batuhanhidiroglu/Desktop/internship/task1/task1/data/weather_data.csv")

df_power["StartDate"] = pd.to_datetime(df_power["StartDate"])
df_weather["Date"] = pd.to_datetime(df_weather["Date"])

df_combined = pd.merge(df_power, df_weather, left_on="StartDate", right_on="Date", how="left")
df_combined.drop(columns=['Date'], inplace=True)
df_combined.to_csv('combined_power_weather_data.csv', index=False)


