""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader


from util import evaluate, load_data


class MLP(nn.Module):

    def __init__(self, d, hidden_d, k, activation = 'relu'):

        super().__init__()

        # two hidden layer
        self.linear1 = nn.Linear(d, hidden_d)
        self.dropout = nn.Dropout(p = 0.2)
        self.linear2 = nn.Linear(hidden_d, k)


    def forward(self, x):

        # pass

        h1 = F.relu(self.linear1(x.float()))
        h1 = self.dropout(h1)
        o = self.linear2(h1)

        return o




class MultilayerPerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self, model, device = 'cpu'):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        # pass

        self.model = model
        self.device = device
        print(self.model)

    def train(self, training_data, learninng_rate = 0.0001, n_epoches = 75, batch_size = 128):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        # TODO: Implement the training of this model.
        # pass

        print('Training')

        # train-val data split
        np.random.seed(1)
        train_val_data = [(np.array(a), b) for (a, b) in training_data]
        np.random.shuffle(train_val_data)

        n = len(train_val_data)

        # 90% - 10% split
        train_data = train_val_data[:int(0.95 * n)]
        val_data = train_val_data[int(0.95 * n):]

        n_train, n_val = len(train_data), len(val_data)

        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False)

        # Initialize the loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learninng_rate)

        # put model on device
        self.model.to(self.device)

        self.best_acc = float('-inf')

        # training loop
        for epoch in range(n_epoches):
            self.model.train()

            # loop over data loader
            for (X, y) in train_loader:


                X = Variable(X).to(self.device)
                y = Variable(y).to(self.device)

                preds = self.model(X)
                loss = criterion(preds, y)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            # Evaluate on the validation dataset
            self.model.eval()

            # set torch to no grad to speed up
            with torch.no_grad():

                for (X, y) in val_loader:

                    X = Variable(X).to(self.device)
                    y = Variable(y).to(self.device)

                    preds = self.model(X)
                    val_loss = criterion(preds, y)

                    _, labels = torch.max(preds, 1)
                    val_acc = (labels == y).sum().item()/(y.size(0))


            print('Epoch: {}, loss {}, val loss {}, val acc {}'.format(epoch, loss, val_loss, val_acc))

            # early stopping




    def predict(self, model_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.
        # return None
        # print('Predicting')

        self.model.eval()
        with torch.no_grad():

            pred = self.model(torch.tensor(model_input))

            _, label = torch.max(pred, 0)

        return int(label)



if __name__ == "__main__":
    train_data, dev_data, test_data, data_type = load_data(sys.argv)
    data_type = 'propername'


    # Try out different hyperparameters
    activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'silu']

    # Train the model using the training data.
    mlp = MLP(d = len(train_data[0][0]), hidden_d = 100, k = 5)
    model = MultilayerPerceptronModel(model = mlp)
    model.train(train_data, n_epoches = 35)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "mlp_" + data_type + "_dev_predictions.csv"))
    print('Dev Acc', dev_accuracy)
    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", "mlp_" + data_type + "_test_predictions.csv"))
