""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np
import torch

from util import evaluate, load_data

class MultilayerPerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        pass

    def train(self, training_data):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        # TODO: Implement the training of this model.
        pass

    def predict(self, model_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.
        return None

if __name__ == "__main__":
    train_data, dev_data, test_data, data_type = load_data(sys.argv)

    # Train the model using the training data.
    model = MultilayerPerceptronModel()
    model.train(train_data)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "mlp_" + data_type + "_dev_predictions.csv"))

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", "mlp_" + data_type + "_test_predictions.csv"))
