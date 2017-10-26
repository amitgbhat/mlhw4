import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('hw2data.mat')
x = np.transpose(data['X'])
y = np.transpose(data['Y'])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def repeat_horizontally(bias, times):
    return np.tile(bias, (1,times))

class MyNeuralNetRestore(object):
    def __init__(self):
        self.weights = [];
        self.biases = [];
        weight_1 = np.load("weights_1.npy");
        weight_0 = np.load("weights_0.npy");
        biases_0 = np.load("biases_0.npy");
        biases_1 = np.load("biases_1.npy");

        self.weights.append(weight_0)
        self.weights.append(weight_1)

        self.biases.append(biases_0);
        self.biases.append(biases_1);

    def predict(self,X):
        weights = self.weights;
        biases = self.biases;

        layer_1 = sigmoid(weights[0].dot(X) + repeat_horizontally(biases[0], X.shape[1]))
        layer_2 = sigmoid(weights[1].dot(layer_1) + repeat_horizontally(biases[1], layer_1.shape[1]))
        return layer_2;

best_neural_net = MyNeuralNetRestore();
predicted = best_neural_net.predict(x);
# print(predicted)

fig = plt.figure(figsize=(8, 6))
plt.scatter(
    x,
    y,
    label="real",
    s = 5.5)
plt.scatter(
    x,
    predicted,
    label="predicted",
    s = 5.5)
plt.legend()
plt.grid()
plt.show()
fig.savefig('approximator.png', dpi=fig.dpi)