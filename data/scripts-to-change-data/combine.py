import pandas as pd

df_power = pd.read_csv("data/prev_versions_data/power_data.csv")
df_weather = pd.read_csv("data/prev_versions_data/weather_data.csv")

df_power["StartDate"] = pd.to_datetime(df_power["StartDate"])
df_weather["Date"] = pd.to_datetime(df_weather["Date"])

df_combined = pd.merge(df_power, df_weather, left_on="StartDate", right_on="Date", how="left")
df_combined.drop(columns=['Date'], inplace=True)
df_combined.to_csv('data/prev_versions_data/combined_power_weather_data.csv', index=False)


