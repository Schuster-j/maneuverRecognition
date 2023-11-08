import math
import random
import numpy as np
from numpy.random import default_rng
import torch
from sklearn.preprocessing import RobustScaler
from torch.autograd import Variable
from typing import Tuple, Any, List


def create_dataset(X: Any, y: Any, time_steps: int = 1, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Function to split data with a categorical y variable into windows with length of given time_steps and an interval
    equal to value of step between the windows. For overlapping windows use a value for step smaller than time_steps.

    :param X: Dataframe with data of predictors.
    :param y: Dataframe column of target variable.
    :param time_steps: Length of windows in number of rows.
    :param step_size: Steps between windows.
    :return: Tuple of numpy array for x-variable data and numpy array for y-variable data."""

    X_windows, y_values = [], []
    for i in range(0, len(X) - time_steps, step_size):
        values = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        X_windows.append(values)
        y_values.append(labels.mode()[0])
    return np.array(X_windows), np.array(y_values).reshape(-1, 1)


def timeseries_train_test_split(data, x_variables, y_variable, splits: int, test_size: float, time_steps: int,
                                step_size: int, scale: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Function to split a timeseries into windowed training and testing data without having data leakage even if
    using overlapping windows.

    :param data: Full dataframe with x and y variables.
    :param x_variables: List of column names of predictor variables.
    :param y_variable: Column name of target variable.
    :param splits: Number of random partitions in which data will be separated before windowing will be applied.
    :param test_size: Proportion of data to use for testing.
    :param time_steps: Length of windows in number of rows.
    :param step_size: Steps between windows.
    :param scale: Apply robust scaling.
    :return: Windowed arrays for training and testing data of predictors and target variable."""

    split_list = np.array_split(data, splits)

    # generate random numbers for selection of data splits
    rng = default_rng()
    random_numbers = rng.choice(splits, size=math.floor(splits * test_size), replace=False)

    df_full = data.copy()
    df_training = data.copy()

    # remove random partitions of data from training data
    for num in random_numbers:
        df_training = df_training.drop(split_list[num].index)

    if scale:
        # scale data based on training data
        scaler = RobustScaler()
        scaler = scaler.fit(df_training[x_variables].values)
        df_training.loc[:, x_variables] = scaler.transform(df_training[x_variables].to_numpy())

    # sample test splits
    X_test, y_test = [], []
    for num in random_numbers:
        split_idx = split_list[num].index

        # get data of partition from full dataframe
        extract = df_full.loc[split_idx, :]

        if scale:
            # apply scaling
            extract.loc[:, x_variables] = scaler.transform(extract[x_variables].to_numpy())

        # apply windowing function on test partitions
        X_slice, y_slice = create_dataset(extract[x_variables], extract[y_variable], time_steps, step_size)

        # append to test data
        X_test.append(X_slice)
        y_test.append(y_slice)

    X_test, y_test = np.vstack(X_test), np.vstack(y_test)

    # apply windowing function on training data
    X_train, y_train = create_dataset(df_training[x_variables], df_training[y_variable], time_steps, step_size)

    return X_train, y_train, X_test, y_test


def check_proportion_value(value: float):
    """ Function to check input of variable proportion_to_remove in function remove_maneuvers."""

    if (value > 1.0) or (value < 0):
        raise ValueError(f'Proportion to remove was set to {value} - has to be between 1.0 and 0.')


def remove_maneuvers(X_train: Any, y_train: Any, X_test: Any, y_test: Any, maneuvers: List[str],
                     proportion_to_remove: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Function to remove maneuvers from given data partitions in a given proportion to balance
    maneuver classes.

    :param X_train: X data of training partition.
    :param y_train: Target variable data of training partition.
    :param X_test: X data of testing partition.
    :param y_test: Target variable data of testing partition.
    :param maneuvers: List of maneuver names or single maneuver name.
    :param proportion_to_remove: Proportion of data to remove for given maneuvers.
    :return: Numpy arrays of X and y data for training and testing partitions.
    """

    check_proportion_value(proportion_to_remove)

    if isinstance(maneuvers, str):
        maneuvers = [maneuvers]

    for maneuver in maneuvers:
        maneuver_train = np.where(y_train == maneuver)[0]
        maneuver_test = np.where(y_test == maneuver)[0]

        # randomly select cases to remove from training data
        remove_maneuver_train = random.sample(list(maneuver_train),
                                              k=round(proportion_to_remove * len(maneuver_train)))

        # randomly select cases to remove from training data
        remove_maneuver_test = random.sample(list(maneuver_test),
                                             k=round(proportion_to_remove * len(maneuver_test)))

        # delete selected cases from data
        X_train = np.delete(X_train, remove_maneuver_train, axis=0)
        y_train = np.delete(y_train, remove_maneuver_train)
        X_test = np.delete(X_test, remove_maneuver_test, axis=0)
        y_test = np.delete(y_test, remove_maneuver_test)

    return X_train, y_train, X_test, y_test


class LabelEncoding:
    """ Class to encode labels from target variable data and transform given data with sklearn LabelEncoder.
    Stores LabelEncoder for usage of inverse_transform(). """

    def __init__(self, y_train: Any, y_test: Any):
        """ Initialize LabelEncoding object based on sklearn LabelEncoder fitted on y training data.
        Attribute self.label_encoder can be used with function inverse_transform to transform y data back to encoded labels.

        :param y_train: Target variable data of training partition.
        :param y_test: Target variable data of testing partition.
        """
        from sklearn.preprocessing import LabelEncoder

        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit_transform(self.y_train)
        self.encoded_labels = self.label_encoder.classes_

    def transform(self):
        """ Function to transform y data by applying label encoding.
        :return: Transformed numpy arrays of training and testing y data.
        """
        self.y_train = self.label_encoder.transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

        return self.y_train, self.y_test


def transform_to_variables(X_train: Any, y_train: Any, X_test: Any, y_test: Any):
    """ Function to transform arrays of training and testing data to torch Variables.

    :param X_train: Numpy array of training predictor data.
    :param y_train: Numpy array of training target variable data.
    :param X_test: Numpy array of testing predictor data.
    :param y_test: Numpy array of training target variable data.
    :return: Data of type torch.Tensor for training and testing partitions.
    """

    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()
    return X_train, y_train, X_test, y_test
