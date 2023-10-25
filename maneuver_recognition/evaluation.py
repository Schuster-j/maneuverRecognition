from typing import List, Tuple, Any

import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np


def relative_predictions(array: Any) -> np.ndarray:
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


def confusion_heatmap(y_test: Any, y_pred: Any, classes: Any, colorscale='Blues', height: int = 900, width: int = 900,
                      relative: bool = False):
    """ Plot heatmap based on confusion matrix of predicted and true values for arbitrary number of classes. Use
    parameter relative to plot values relative to the predictions of every single class.

    :param y_test: True class values of test data.
    :param y_pred: Predicted class values of test data.
    :param classes: Class labels.
    :param colorscale: Plotly colorscale.
    :param height: Figure height.
    :param width: Figure width.
    :param relative: Calculate the proportion of correctly classified values relative to each class separately.
    :return: Plotly figure.
    """

    from sklearn.metrics import confusion_matrix

    title = 'Confusion Heatmap'
    confusion_matrix = confusion_matrix(y_test, y_pred)

    if relative:
        title = 'Confusion Heatmap (relative values)'
        confusion_matrix = np.apply_along_axis(relative_predictions, 1, confusion_matrix)

    heatmap = go.Heatmap(z=confusion_matrix, x=classes, y=classes, colorscale=colorscale)  # FEHLER IRGENDWO

    fig = go.Figure(data=[heatmap])

    fig.update_layout(title= title, yaxis=dict(categoryorder='category descending'))
    fig.update_yaxes(title_text="Actual")
    fig.update_xaxes(title_text="Predicted")

    fig.layout.height = height
    fig.layout.width = width

    return fig
