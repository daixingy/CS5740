from perceptron import PerceptronModel
from MLP_Name import MLP, MultilayerPerceptronModel
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# X = [(np.array([1,1,1]), 1)]
# X.append((np.array([1, -1, 1]), 1))
# X.append((np.array([-1, -1, 1]), 0))
# X.append((np.array([0.25, 0.25, 1]), 0))

# model = PerceptronModel()

# model.train(X)

# test_x = np.array([-10, 10, 1])
# print(model.predict(test_x))

# a = [([1,2,3], 1), ([2,3,4], 2), ([3,4,5], 3)]
# b = np.array(a)

# dataset = [(np.array([1,2,3]), 1), (np.array([2,3,4]), 2), (np.array([4,5,6]), 7)]
# loader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True)

# for (X, y) in loader:
# 	print(X.float())
# 	print(y)
a = ([1,2,3], [2,3,4], [2,2,2], [3,3,3], [4,4,4])
b = ([1,2,3,4,5])

print(list(zip(a, b)))
