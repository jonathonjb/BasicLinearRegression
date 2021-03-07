import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def createData(jitter = 0.5):
    # data = np.sin(np.arange(0, 20, 0.1)) * (((np.random.rand(200) - 0.5) * jitter) + 1)
    X = np.arange(0, 100, 1).reshape((100, 1))
    y = np.arange(0, 100) * 0.1 + 5
    return X, y

def plotData(X, y):
    plt.plot(X, y)
    plt.ylim(0, 20)
    plt.show()

def main():
    X, y = createData()
    plotData(X, y)
    model = LinearRegression(numOfParameters=1)
    history = model.train(X=X, y=y, learningRate=.0001,numOfRounds=100)

    # plt.plot(history)
    # plt.show()

    predictions = model.getPredictions(X)
    plotData(X, predictions)



if __name__ == '__main__':
    main()