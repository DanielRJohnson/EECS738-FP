import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from itertools import combinations


def add_deltas_and_time(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Taken and refactored from Bailey's notebook
    Given the data in dataframe form, this function returns a new dataframe
    with differentials added and a time index
    """
    df = data_df.copy()

    for feature in ["AirTemp", "QV", "Omega", "SeaLevPress", "UWinds", "VWinds", "Lat", "Lon"]:
        df["d" + feature] = df[feature].diff()  # Add differential
        df.loc[0, "d" + feature] = 0  # original differential is zero

    df['Time'] = pd.to_datetime(df.Time, format="%Y%m%d%H")
    df = df.set_index('Time')  # make the index time

    return df


def add_extreme(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Taken and refactored from Bailey's notebook
    Given the data in dataframe form, this function returns a new dataframe
    with a boolean extreme value based on our definition of extreme
    """
    df = data_df.copy()

    LQ = df.LH.quantile(0.25)
    UQ = df.LH.quantile(0.75)
    bound = 1.5 * (UQ - LQ)  # Whisker length * IQR
    lower_bound = LQ - bound
    upper_bound = UQ + bound
    df["Extreme"] = np.where((df["LH"] > upper_bound) | (df["LH"] < lower_bound), 1, 0)

    return df


def scale_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Taken and refactored from Bailey's notebook
    Given the data in dataframe form, this function returns a new dataframe
    with the data scaled with sklearn.preprocessing.RobustScaler
    """
    df = data_df.copy()
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


def column_combinations(columns: list[str], max_len: int) -> list[list[str]]:
    """
    Taken and refactored from Bailey's notebook
    Given a list of columns and a maximum length, this function returns all
    combinations of the given columns of the maximum length or less
    """
    combs = []
    for i in range(1, max_len+1):
        combs.extend([list(comb) for comb in combinations(columns, i)])
    return combs
