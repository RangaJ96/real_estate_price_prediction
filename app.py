import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20, 10)
import util


class DataProcessor(util.Utilities):
    Util = util.Utilities

    def read_data_set(self):
        return pd.read_csv(self.dataPath)

    def remove_outliers(self,dataFrame):
        print(f'print the given data frame to remove outliers : \n {dataFrame.head(10)}')
        '''\n The Business Logic that was applied here is :
                A bedroom can not be size of less than 300 sqft
                *so every data element which can not meet that criteria will be removed\n'''
        df = dataFrame[~(dataFrame.total_sqft/dataFrame.bhk < 300)]
        print(f'Data Frame after removing the outliers \n {df.head(10)}\n {df.shape}\n')

    def feature_engineering(self, dataFrame):
        df = dataFrame.copy()
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        print(f'\n\n After adding the price per sqrt property \n\n {df.head()}')
        print(f'\n Unique location stat : {len(df.location.unique())}\n')
        df.location = df.location.apply(lambda x: x.strip())
        location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
        print(f'\n {location_stats}\n')

        # group lower observation to a common category
        location_stat_less_than_ten = location_stats[location_stats <= 10]
        print(f'\n{location_stat_less_than_ten}\n')
        df.location = df.location.apply(lambda x: 'other' if x in location_stat_less_than_ten else x)
        return df

    def data_pre_process(self):
        df1 = self.read_data_set()
        # print relevant data fields get an idea about the data set
        print(f'Print the number of columns and rows of the given data-set {df1.shape}')

        # removing the unnecessary data fields
        df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
        print(
            f'\n\nFirst five rows of the data set after removing all the unnecessary data fields \n\n {df2.head()}\n\n ')

        # data cleaning
        df2.isnull().sum()
        # drop the NaN columns
        df3 = df2.dropna()
        df3.isnull().sum()
        print(f'\nNumber of columns and rows after removing NaN contents {df3.shape}\n')
        print(df3['size'].unique())
        df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

        # print the unique BHK values to find the abnormal values
        print(df3['bhk'].unique())
        print(df3[df3.bhk > 20])

        # Data cleaning : remove unstructured data
        print(df3[~df3['total_sqft'].apply(self.Util.is_float)].head(), ' \n')

        df4 = df3.copy()
        df4['total_sqft'] = df4['total_sqft'].apply(self.Util.covert_sqft_to_num)
        print(df4.head(10))

        # Optimize the data set
        df5 = self.feature_engineering(df4)

        # remove outliers
        self.remove_outliers(df5)

    def __init__(self, dataPath, util):
        self.dataPath = dataPath
        self.util = util


if __name__ == "__main__":
    Util = util.Utilities()
    DP = DataProcessor("Bengaluru_House_Data.csv", Util)
    DP.data_pre_process()
