import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import json
plt.style.use('ggplot')

def main():
    df = pd.read_csv('data/data_v3.csv')
    correlation_result = correlation(df)
    single_regression_result = single_regression(df)

    with open('correlation_result.json', 'w') as f:
        json.dump(correlation_result, f)
    

def correlation(dataframe):
    df_corr = dataframe[['day_of_week_x', 'Hour_of_Day', 'Value (kWh)', 'Temp_avg', 'HDD', 'CDD']].dropna().corr()
    print(df_corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')

    plt.savefig('correlation_matrix.png')
    plt.close()

    return {'correlation_matrix': str(df_corr)}

def single_regression(dataframe):
    sns.pairplot(dataframe,
                 vars=['day_of_week_x', 'Hour_of_Day', 'Value (kWh)', 'Temp_avg', 'HDD', 'CDD'],
                 hue='notes')
    plt.savefig('single_regression_graphs.png')
    plt.close()
    return {'single_regression_graphs': 'single_regression_graphs.png'}

if __name__ == "__main__":
    main()


