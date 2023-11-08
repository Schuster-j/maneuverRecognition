import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


class ManeuverModel(nn.Module):
    """ Class with LSTM based model architecture for maneuver recognition."""

    def __init__(self, n_features: int, n_classes: int, n_hidden: int = 24,
                 n_layers: int = 4, lstm_dropout: float = 0.7,
                 n_features_linear1: int = 64, n_features_linear2: int = 32,
                 linear_dropout: float = 0.3):
        """ Initialization of model architecture.

        :param n_features: The number of expected features in the input x.
        :param n_classes: Number of classes for classification layer.
        :param n_hidden: Number of features in the hidden state of the LSTM.
        :param n_layers: Number of stacked LSTM layers.
        :param lstm_dropout: Value of applied dropout in LSTM layers.
        :param n_features_linear1: Number of features in first linear layer.
        :param n_features_linear2: Number of features in second linear layer.
        :param linear_dropout: Value of applied dropout between first and second linear layer.
        """

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        self.full_layer1 = nn.Linear(n_hidden, n_features_linear1)
        self.dropout = nn.Dropout(linear_dropout)
        self.full_layer2 = nn.Linear(n_features_linear1, n_features_linear2)
        self.classifier = nn.Linear(n_features_linear2, n_classes)

    def forward(self, x):
        """ Forward propagation. """
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]
        out = self.dropout(F.relu(self.full_layer1(out)))
        out = F.relu(self.full_layer2(out))
        out = self.classifier(out)
        return out

    def predict(self, X):
        """ Function to use model for prediction of given cases.

        :param X: X data to use for prediction.
        :return: List of predictions for given input data.
        """

        return [pred.argmax() for pred in self(X)]


def train(dataloader, model, loss_fn, optimizer, device):
    """ Function to apply training process on model with given data of dataloader object.
    In order to fit the model with direct data use fit_model.

    :param dataloader: Dataloader object for training data.
    :param model: Model object.
    :param loss_fn: Loss function.
    :param optimizer: Optimizer object for optimization algorithm.
    :param device: Device to use.
    """

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)


def test(dataloader, model, loss_fn, device):
    """ Function to evaluate given model with data of dataloader. In order to use the model for predictions use
    the predict function of the model object instead.

    :param dataloader: Dataloader object for test data.
    :param model: Model object.
    :param loss_fn: Loss function.
    :param device: Device to use.
    :return: Tuple with proportion of correctly predicted cases and average test loss.
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return [correct, test_loss]


def fit_model(model, X_train, y_train, X_test, y_test, epochs, batch_size,
                         loss_function, optimizer, device):
    """ Function to fit a model. Applies model training with given X and y training data and uses given X and y test
    data for training validation. Returns list of validation loss and validation accuracy per epoch.

    :param model: Model object
    :param X_train: X data of training partition.
    :param y_train: Target variable data of training partition.
    :param X_test: X data of testing partition.
    :param y_test: Target variable data of testing partition.
    :param epochs: Number of epochs in training process.
    :param batch_size: Batch size to use for training and testing.
    :param loss_function: Loss function.
    :param optimizer: Optimizer object for optimization algorithm.
    :param device: Device to use.
    :return: List of validation loss and validation accuracy values for every epoch.
    """

    train_dataloader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size)
    test_dataloader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size)

    loss_list = np.zeros((epochs,))
    accuracy_list = np.zeros((epochs,))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer, device)
        accuracy, test_loss = test(test_dataloader, model, loss_function, device)
        accuracy_list[epoch] = accuracy
        loss_list[epoch] = test_loss
    print("Done!")

    return loss_list, accuracy_list
