""" Maximum entropy model for Assignment 1: Starter code.
You can change this code however you like. This is just for inspiration.
"""
import os
import sys
import numpy as np

from util import evaluate, load_data, compute_accuracy

class PerceptronModel():
    """ Maximum entropy model for classification.
    Attributes:
    """
    def __init__(self, lr = 0.01, max_iters = 20, k = 2):
        '''Initialize the perceptron model
        Inputs:
            lr: learning rate
            max_iters: maximum iterations allowed
            k: number of classes        
        '''

        self.lr = lr
        self.max_iters = max_iters
        self.k = k



    def train(self, training_data):
        """ Trains the maximum entropy model.
        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        # TODO: Implement the training of this model.
        # pass

        # shuffle the dataset, train-val split
        np.random.seed(1)
        n = len(training_data)

        train_val_data = training_data
        np.random.shuffle(train_val_data)

        training_data = train_val_data[:int(0.9 * n)]
        val_data = train_val_data[int(0.9 * n):]
        val_features, val_labels = zip(*val_data)
        val_features, val_labels = np.c_[np.array(val_features), np.ones(len(val_features))], np.array(val_labels)

        n_train, n_val = len(training_data), len(val_data)
        print('training samples, validation samples', n_train, n_val)
        # initialize weights
        d = len(training_data[0][0])
        self.W = np.zeros((self.k, d + 1))

        iters = 0

        while True:


            # initialize error count
            error_count = 0
            i = 0

            # shuffle training data
            np.random.shuffle(training_data)

            # loop over training data
            for (X, y) in training_data:

                X = np.r_[np.array(X), np.ones(1)]
                # print(self.W.shape, X.shape)
                pred = int(np.argmax(self.W @ X, axis = 0))

                # update
                if pred != y:

                    self.W[pred] = self.W[pred] - self.lr * X
                    self.W[y] = self.W[y] + self.lr * X

                    error_count += 1

            # validate on the validation dataset
            val_acc = compute_accuracy(val_labels, np.argmax(val_features @ self.W.T, axis = 1))


            iters += 1

            # break if no errors
            if error_count == 0 or iters >= self.max_iters:
                break

            # if (iters + 1) % 20 == 0:
            print('Epoch {}, train acc {}, val acc {}'.format(iters, 1 - error_count/n_train, val_acc))

        
    def predict(self, model_input):
        """ Predicts a label for an input.
        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector. (d, )
        Returns:
            The predicted class: int
        """
        # TODO: Implement prediction for an input.
        # return None

        # preds = np.argmax(model_input @ self.W.T, axis = 1).squeeze()

        pred = int(np.argmax(self.W @ np.r_[np.array(model_input), np.ones(1)], axis = 0))

        return pred




if __name__ == "__main__":
    train_data, dev_data, test_data, data_type = load_data(sys.argv)

    # print(type(train_data[0][0]))


    print('training length {}, dev length {}, test length {}'.format(len(train_data), len(dev_data), len(test_data)))

    # Train the model using the training data.
    model = PerceptronModel(k = 5)
    model.train(train_data)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "perceptron_" + data_type + "_dev_predictions.csv"))
    print('Dev Acc', dev_accuracy)
    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", "perceptron_" + data_type + "_test_predictions.csv"))