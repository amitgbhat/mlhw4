import numpy as np
import scipy.io as sio

# getting random number from unit gaussian
def get_random(rows, cols):
    return np.random.normal(0,1,(rows, cols));

#get layer weight matrices.
def get_layers(input_dimension, layers, output_dimension):
    
    prev_layer = input_dimension;
    weights = [];
    biases = [];

    for layer in layers:
        weight = get_random(layer, prev_layer);
        bias = get_random(layer,1);
        weights.append(weight);
        biases.append(bias);
        prev_layer = layer;

    weight = get_random(output_dimension, prev_layer);
    bias = get_random(output_dimension,1);
    biases.append(bias);
    weights.append(weight);
    return weights, biases;

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def repeat_horizontally(bias, times):
    return np.tile(bias, (1,times))

# Load data from the mat files.
data = sio.loadmat('hw2data.mat')
x = np.transpose(data['X'])
y = np.transpose(data['Y'])

# Note that each row is a feature and each column is a sample.

input_dimension = x.shape[0];
layers = [100]
output_dimension = y.shape[0];
learning_rate = 0.0001;

class MyNeuralNet(object):
    def __init__(self):
        self.input_dimension = input_dimension;
        self.layers = layers
        self.output_dimension = output_dimension;
        self.learning_rate = learning_rate;
        self.weights, self.biases = get_layers(self.input_dimension, self.layers, self.output_dimension);
    
    def partial_fit(self,X,Y):

        # Compute the forward pass.
        sigma_1 = sigmoid(self.weights[0].dot(X) + repeat_horizontally(self.biases[0], X.shape[1]))
        sigma_2 = sigmoid(self.weights[1].dot(sigma_1) + repeat_horizontally(self.biases[1], sigma_1.shape[1]))
        delta = sigma_2 - y;

        # Compute the backward pass with the gradients.
        # Reshape bias gradient to have an explicit 1 column.            
        d_error_by_d_b2 = np.sum((delta * sigma_2 * (1 - sigma_2)), axis=1).reshape(self.biases[1].shape[0],1);
        d_error_by_d_w2 = (delta * sigma_2 * (1 - sigma_2)).dot(np.transpose(sigma_1));

        # Reshape bias gradient to have an explicit 1 column.
        d_error_by_d_b1 = np.sum(((np.transpose(self.weights[1])).dot(delta * sigma_2 * (1 - sigma_2))) * sigma_1 * (1-sigma_1), axis=1).reshape(self.biases[0].shape[0],1)
        d_error_by_d_w1 = (((np.transpose(self.weights[1])).dot(delta * sigma_2 * (1 - sigma_2))) * sigma_1 * (1-sigma_1)).dot(np.transpose(X))

        # Compute current cost 
        delta_squared = delta * delta;
        current_cost = np.mean(delta_squared) / 2;

        # Now that we have a gradient, we can see if the current learning rate is going to overshoot the
        # optima. If so, we need to reduce the learning rate so that after proceeding in the gradient direction
        # the cost actually decreases.
        # The other part of increasing the weight if we are not making much change to cost after gradient descent
        # is handled in training method (should have been here).
        # Doing this 50 times due to computational time constraint.
        for i in range(50):
            temp_weight_0 = self.weights[0] - (self.learning_rate * d_error_by_d_w1);
            temp_weight_1 = self.weights[1] - (self.learning_rate * d_error_by_d_w2);
            temp_bias_0 = self.biases[0] - (self.learning_rate * d_error_by_d_b1);
            temp_bias_1 = self.biases[1] - (self.learning_rate * d_error_by_d_b2);

            temp_sigma_1 = sigmoid(temp_weight_0.dot(X) + repeat_horizontally(temp_bias_0, X.shape[1]))
            temp_sigma_2 = sigmoid(temp_weight_1.dot(temp_sigma_1) + repeat_horizontally(temp_bias_1, temp_sigma_1.shape[1]))
            temp_delta = temp_sigma_2 - y;
            temp_delta_squared = temp_delta * temp_delta;
            temp_cost = np.mean(temp_delta_squared) / 2;

            # if the costs are overshooting optima. Decrease learning rate.
            if temp_cost > current_cost:
                self.learning_rate = self.learning_rate * 0.99;
            else:
                break;
            
        # Update the weights.
        self.weights[0] = self.weights[0] - (self.learning_rate * d_error_by_d_w1);
        self.weights[1] = self.weights[1] - (self.learning_rate * d_error_by_d_w2);
        self.biases[0] = self.biases[0] - (self.learning_rate * d_error_by_d_b1);
        self.biases[1] = self.biases[1] - (self.learning_rate * d_error_by_d_b2);

        # Compute the new cost and return.
        sigma_1 = sigmoid(self.weights[0].dot(X) + repeat_horizontally(self.biases[0], X.shape[1]))
        sigma_2 = sigmoid(self.weights[1].dot(sigma_1) + repeat_horizontally(self.biases[1], sigma_1.shape[1]))
        delta = sigma_2 - y;

        delta_squared = delta * delta;
        return np.mean(delta_squared) / 2;

    def predict(self,X):
        weights = self.weights;
        biases = self.biases;

        layer_1 = sigmoid(weights[0].dot(X) + repeat_horizontally(biases[0], X.shape[1]))
        layer_2 = sigmoid(weights[1].dot(layer_1) + repeat_horizontally(biases[1], layer_1.shape[1]))
        delta = layer_2 - y;

        delta_squared = delta * delta;
        return layer_2, np.mean(delta_squared) / 2;

iterations = 1000;
max_cost_for_convergence = 0.0001;

number_of_networks = 10;
min_cost = 100;

for j in range(number_of_networks):
        
    neural_net = MyNeuralNet();
    prev_cost = neural_net.partial_fit(x, y)
    for i in range(iterations):
        
        cost = neural_net.partial_fit(x, y)

        # We see that we are not making much progress in the gradient descent step.
        # so we increase the learning rate. The handling for auto decreasing learning rate is present in the
        # class definition.
        if prev_cost - cost < 0.0001:
            neural_net.learning_rate = neural_net.learning_rate * 1.005;  

        # We can stop when we have hit a good cost.
        if cost < max_cost_for_convergence:
            print('Cost {0} is less than max allowed cost {1}. Breaking'.format(cost, max_cost_for_convergence));
            break;

        prev_cost = cost;

    predicted,cost = neural_net.predict(x);

    # Keep updating the best neural network.
    if cost < min_cost:
        print('Iteration #{0}: Updating best neural network with cost: {1}'.format(j+1, cost))
        min_cost = cost;
        best_neuralnet = neural_net;
    else:
        print('Iteration #{0}: Current neural network has cost {1} which is greater than best neural net {2}'.format(j+1, cost, min_cost))
    

predicted,cost = best_neuralnet.predict(x);
print('predicted & real', predicted, y)
print("cost",cost);

# Save the weights so that it can be used later for prediction.
np.save("weights_0", best_neuralnet.weights[0])
np.save("weights_1", best_neuralnet.weights[1])
np.save("biases_0", best_neuralnet.biases[0])
np.save("biases_1", best_neuralnet.biases[1])