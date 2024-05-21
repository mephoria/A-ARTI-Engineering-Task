import pandas as pd

df = pd.read_csv('data/prev_versions_data/final_finaldata.csv')
df = df.drop(['hdd', 'cdd'], axis=1)
df['StartDate'] = pd.to_datetime(df['StartDate'])


def calculate_hdd_cdd(temp_avg):
    base_temp = 65
    hdd = max(0, base_temp - temp_avg)
    cdd = max(0, temp_avg - base_temp)
    return hdd, cdd


df['HDD'], df['CDD'] = zip(*df.apply(lambda row: calculate_hdd_cdd(row['Temp_avg']), axis=1))
df['HDD'] = df.groupby(df['StartDate'].dt.date)['HDD'].ffill()
df['CDD'] = df.groupby(df['StartDate'].dt.date)['CDD'].ffill()


df.to_csv('absolute_data.csv', index=False)


