import numpy as np
import random

class LinearRegression:
    def __init__(self, numOfParameters):
        self.initializeParameters(numOfParameters)

    def initializeParameters(self, numOfParameters):
        self.weights = np.random.rand(numOfParameters)
        self.bias = random.random() * 5

    def train(self, X, y, learningRate=0.1, numOfRounds=100):
        history = []
        for i in range(numOfRounds):
            predictions = self.getPredictions(X)

            cost = self.getCost(predictions, y)
            history.append(cost)
            print(i, '-', cost)

            weightsGradient = 1 / X.shape[0] * np.dot(X.T, predictions - y)

            self.weights -= learningRate * weightsGradient
            self.bias -= learningRate * (np.sum(weightsGradient))

        return np.array(history)

    def getPredictions(self, X):
        return np.dot(X, self.weights) + self.bias

    def getCost(self, predictions, y):
        return 1/y.shape[0] * np.sum(np.square(y - predictions))