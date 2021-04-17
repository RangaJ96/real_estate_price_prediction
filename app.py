import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

matplotlib.rcParams["figure.figsize"] = (20, 10)
import util
import pickle
import json


class DataProcessor(util.Utilities):
    Util = util.Utilities

    def read_data_set(self):
        return pd.read_csv(self.dataPath)

    def model_training(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        lr_clf = LinearRegression()
        lr_clf.fit(X_train, y_train)
        lr_clf.score(X_test, y_test)

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        print(cross_val_score(LinearRegression(), X, y, cv=cv))
        return lr_clf

    def find_best_model_using_gridsearchcv(self, X, y):
        algos = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {
                    'normalize': [True, False]
                }
            },
            'lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [1, 2],
                    'selection': ['random', 'cyclic']
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(),
                'params': {
                    'criterion': ['mse', 'friedman_mse'],
                    'splitter': ['best', 'random']
                }
            }
        }
        scores = []
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        for algo_name, config in algos.items():
            gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
            gs.fit(X, y)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
            })

        return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

    def predict_price(self, X, model, location, sqft, bath, bhk):
        loc_index = np.where(X.columns == location)[0][0]

        x = np.zeros(len(X.columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        return model.predict([x])[0]

    def remove_bhk_outliers(self, df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices,
                                                bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    def data_visualization(self, dataframe, location):
        bhk2 = dataframe[(dataframe.location == location) & (dataframe.bhk == 2)]
        bhk3 = dataframe[(dataframe.location == location) & (dataframe.bhk == 3)]
        matplotlib.rcParams['figure.figsize'] = (15, 10)
        plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
        plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
        plt.xlabel("Total Square Feet Area")
        plt.ylabel("Price (Lakh LKR)")
        plt.title(location)
        plt.legend()

    def remove_outliers(self, dataFrame):
        print(f'print the given data frame to remove outliers : \n {dataFrame.head(10)}')
        '''\n The Business Logic that was applied here is :
                A bedroom can not be size of less than 300 sqft
                *so every data element which can not meet that criteria will be removed\n'''
        df = dataFrame[~(dataFrame.total_sqft / dataFrame.bhk < 300)]
        print(f'Data Frame after removing the outliers \n {df.head(10)}\n {df.shape}\n')
        return self.Util.remove_pps_outliers(df)

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
        df6 = self.remove_outliers(df5)

        self.data_visualization(df6, "Rajaji Nagar")

        df7 = self.remove_bhk_outliers(df6)

        df8 = df7[df7.bath < df7.bhk + 2]

        df9 = df8.drop(['size', 'price_per_sqft'], axis='columns')

        # one hot encoding for the location
        dummies = pd.get_dummies(df9.location)

        df10 = pd.concat([df9, dummies.drop('other', axis='columns')], axis='columns')

        df11 = df10.drop('location', axis='columns')

        # indepedent variable set
        x = df11.drop('price', axis='columns')

        # dependent price variable
        y = df11.price

        trained_model = self.model_training(x, y)

        # self.find_best_model_using_gridsearchcv(x, y)

        print(self.predict_price(x, trained_model, '1st Phase JP Nagar', 1000, 2, 2))

        with open('home_prices_model.pickle', 'wb') as f:
            pickle.dump(trained_model, f)

        columns = {
            'data_columns': [col.lower() for col in x.columns]
        }
        with open("columns.json", "w") as f:
            f.write(json.dumps(columns))

    def __init__(self, dataPath, util):
        self.dataPath = dataPath
        self.util = util


if __name__ == "__main__":
    Util = util.Utilities()
    DP = DataProcessor("House_Data.csv", Util)
    DP.data_pre_process()
