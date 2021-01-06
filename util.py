import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


class Utilities:

    def is_float(x):
        try:
            float(x)
        except:
            return False
        return True

    def covert_sqft_to_num(x):
        token = x.split('-')
        if len(token) == 2:
            return (float(token[0]) + float(token[1]))/2
        try:
            return float(x)
        except:
            return None

    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            mean = np.mean(subdf.price_per_sqft)
            sd = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (mean-sd)) & (subdf.price_per_sqft <= (mean+sd))]
            df_out = pd.concat([df_out,reduced_df],ignore_index=True)
        return df_out
