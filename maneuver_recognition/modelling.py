import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


class ManeuverModel(nn.Module):
    """ Class with prototypical LSTM based model architecture for maneuver recognition."""

    def __init__(self, n_features: int, n_classes: int, n_hidden: int = 24, n_layers: int = 4,
                 lstm_dropout: float = 0.7):
        """ Initialization of model architecture.

        :param n_features: The number of expected features in the input x.
        :param n_classes: Number of classes for classification layer.
        :param n_hidden: The number of features in the hidden state.
        :param n_layers: Number of stacked LSTM layers.
        :param lstm_dropout: Rate of applied dropout in LSTM layers.
        """

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,  # Format
            dropout=lstm_dropout
        )

        self.full_layer1 = nn.Linear(n_hidden, 64)
        self.dropout = nn.Dropout(0.3)
        self.full_layer2 = nn.Linear(64, 32)

        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        """ Forward propagation. """
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]
        out = self.dropout(F.relu(self.full_layer1(out)))
        out = F.relu(self.full_layer2(out))
        out = self.classifier(out)
        return out


def train(dataloader, model, loss_fn, optimizer, device):
    """ Internal training function for given model.
    For training the model with data use train_maneuver_model.

    :param dataloader: Dataloader object for training data.
    :param model: Model object.
    :param loss_fn: L.oss function.
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
    """ Internal test function for given model.

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


def train_maneuver_model(model, X_train, y_train, X_test, y_test, epochs, batch_size,
                         loss_function, optimizer, device):
    """ Function to apply train model with given X and y training data. Uses given X and y test data
    for training validation.

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
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer, device)
        accuracy, test_loss = test(test_dataloader, model, loss_function, device)
        accuracy_list[epoch] = accuracy
        loss_list[epoch] = test_loss
    print("Done!")

    return loss_list, accuracy_list


def predict(X_test, model):
    """ Function to use model for prediction of given cases.

    :param X_test: X data to use for prediction.
    :param model: Model object to perform prediction.
    :return: List of predictions for given input data.
    """

    return [pred.argmax() for pred in model(X_test)]
