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

    def __init__(self, d , k, activation = 'relu'):

        super().__init__()

        # two hidden layer
        self.linear1 = nn.Linear(d, 100)
        self.drop1 = nn.Dropout(p = 0.2)
        self.linear2 = nn.Linear(100, 64)
        self.drop2 = nn.Dropout(p = 0.2)
        self.linear3 = nn.Linear(64, k)


    def forward(self, x):

        # pass

        h1 = F.relu(self.linear1(x.float()))
        h1 = self.drop1(h1)
        h2 = F.relu(self.linear2(h1))
        h2 = self.drop2(h2)
        o = self.linear3(h2)

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

    def train(self, training_data, learninng_rate = 0.0001, n_epoches = 75, batch_size = 256):
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

    data_type = 'newsgroup'
    train_data, dev_data, test_data, labels_dict = load_data(sys.argv)
    k = len(set(train_data[1]))
    print(k)
    train_data = list(zip(train_data[0], train_data[1]))
    dev_data = list(zip(dev_data[0], dev_data[1]))
    test_data = list(zip(test_data[0], test_data[1]))


    # Train the model using the training data.
    mlp = MLP(d = len(train_data[0][0]), k = k)
    model = MultilayerPerceptronModel(model = mlp)
    model.train(train_data, n_epoches = 100)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "mlp_" + data_type + "_dev_predictions.csv"), labels_dict = labels_dict)
    print('Dev Acc', dev_accuracy)
    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", "mlp_" + data_type + "_test_predictions.csv"), labels_dict = labels_dict)