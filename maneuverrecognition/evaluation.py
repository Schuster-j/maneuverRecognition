from typing import List, Tuple, Any
from sklearn.metrics import confusion_matrix as cm
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np


def relative_values(array: Any) -> np.ndarray:
    """ Function to calculate values relative to sum of array. """

    return array / sum(array)


def plot_training_process(loss_list: List, accuracy_list: List, figsize: Tuple = (12, 6)):
    """ Plot validation accuracy and validation loss over epochs.

    :param loss_list: List of validation loss by epoch.
    :param accuracy_list: List of validation loss by epoch.
    :param figsize: Tuple of figure size.
    """

    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize, sharex=True)

    ax1.plot(accuracy_list)
    ax1.set_ylabel("validation accuracy")
    ax2.plot(loss_list)
    ax2.set_ylabel("validation loss")
    ax2.set_xlabel("epochs");


def create_heatmap(data: Any, classes: Any, colorscale: Any, title: str, height: int, width: int,
                   x_label: str = 'Predicted',
                   y_label: str = 'Actual'):
    """ Create a heatmap plot of given data and class labels.

    :param data: Heatmap values.
    :param classes: Class labels.
    :param colorscale: Plotly continuous colorscale.
    :param title: Figure title.
    :param height: Figure height.
    :param width: Figure width.
    :return: Plotly figure.
    """

    fig = px.imshow(data,
                    labels=dict(x=x_label, y=y_label),
                    x=classes,
                    y=classes,
                    color_continuous_scale=colorscale)

    fig.update_traces(text=data, texttemplate="%{text:.2f}")
    fig.update_layout(title_text=title)
    fig.layout.height = height
    fig.layout.width = width

    return fig


def confusion_heatmap(y_test: Any, y_pred: Any, classes: Any, colorscale='Blues', height: int = 900, width: int = 900,
                      title: str = 'Confusion heatmap'):
    """ Create heatmap plot based on confusion matrix of predicted and true values for arbitrary number of classes. Uses create_heatmap() function for plot generation.

    :param y_test: Actual class values of test data.
    :param y_pred: Predicted class values of test data.
    :param classes: Class labels.
    :param colorscale: Plotly colorscale.
    :param height: Figure height.
    :param width: Figure width.
    :param title: Figure title.
    :return: Plotly figure.
    """
    confusion_matrix = cm(y_test, y_pred)
    fig = create_heatmap(confusion_matrix, classes, colorscale, title, height, width)

    return fig


def precision_heatmap(y_test: Any, y_pred: Any, classes: Any, colorscale='Blues',
                      height: int = 900, width: int = 900,
                      title: str = 'Precision heatmap (column wise relative values)'):
    """ Create heatmap plot based on confusion matrix of predicted and true values with relative values along axis of each
    predicted class (column wise). This represents a precision value for each class in the diagonal of the heatmap. Uses create_heatmap() function for plot generation.

    :param y_test: Actual class values of test data.
    :param y_pred: Predicted class values of test data.
    :param classes: Class labels.
    :param colorscale: Plotly colorscale.
    :param height: Figure height.
    :param width: Figure width.
    :param title: Figure title.
    :return: Plotly figure.
    """

    confusion_matrix = cm(y_test, y_pred)
    confusion_matrix = np.apply_along_axis(relative_values, 0, confusion_matrix)
    fig = create_heatmap(confusion_matrix, classes, colorscale, title, height, width)

    return fig


def recall_heatmap(y_test: Any, y_pred: Any, classes: Any, colorscale='Blues',
                   height: int = 900, width: int = 900, title: str = 'Recall heatmap (row wise relative values)'):
    """ Create heatmap plot based on confusion matrix of predicted and true values with relative values along axis of each
    predicted class (row wise). This represents a recall value for each class in the diagonal of the heatmap. Uses create_heatmap() function for plot generation.

    :param y_test: Actual class values of test data.
    :param y_pred: Predicted class values of test data.
    :param classes: Class labels.
    :param colorscale: Plotly colorscale.
    :param height: Figure height.
    :param width: Figure width.
    :param title: Figure title.
    :return: Plotly figure.
    """
    confusion_matrix = cm(y_test, y_pred)
    confusion_matrix = np.apply_along_axis(relative_values, 1, confusion_matrix)
    fig = create_heatmap(confusion_matrix, classes, colorscale, title, height, width)

    return fig
