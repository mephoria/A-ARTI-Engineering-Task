import pandas as pd
import numpy as np

df = pd.read_csv("data/prev_versions_data/combined_power_weather_data.csv")
base_temp = 65 #In Fahrenheit


def get_hdd(val, temp):
    if val < temp:
        return temp - val
    return np.nan

def get_cdd(val, temp):
    if val > temp:
        return val - temp
    return np.nan



df["hdd"] = df["Temp_avg"].apply(lambda x: get_hdd(x, base_temp) if pd.notna(x) else np.nan)
df["cdd"] = df["Temp_avg"].apply(lambda x: get_cdd(x, base_temp) if pd.notna(x) else np.nan)



df.to_csv("data/prev_versions_data/final_data.csv", index=False)

