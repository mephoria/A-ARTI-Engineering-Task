import pandas as pd

main_df = pd.read_csv('data/prev_versions_data/data_v2.csv')
notes_df = pd.read_csv('data/prev_versions_data/power_data.csv')

main_df['StartDate'] = pd.to_datetime(main_df['StartDate'])
notes_df['StartDate'] = pd.to_datetime(notes_df['StartDate'])

merged_df = pd.merge(main_df, notes_df[['StartDate', 'notes']], on='StartDate', how='left')

merged_df.to_csv('data/data_v3.csv', index=False)

