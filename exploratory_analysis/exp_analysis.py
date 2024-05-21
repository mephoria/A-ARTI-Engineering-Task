import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')

# correlation and single_regression function can not run at the same time
def main():
    df = pd.read_csv('data/data_v3.csv')
    # correlation(df)
    single_regression(df)

def correlation(dataframe):
    df_corr = dataframe[['day_of_week_x', 'Hour_of_Day', 'Value (kWh)', 'Temp_avg', 'HDD', 'CDD']].dropna().corr()
    print(df_corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

def single_regression(dataframe):
    sns.pairplot(dataframe,
                 vars=['day_of_week_x', 'Hour_of_Day', 'Value (kWh)', 'Temp_avg', 'HDD', 'CDD'],
                 hue='notes')
    plt.show()



if __name__ == "__main__":
    main()


