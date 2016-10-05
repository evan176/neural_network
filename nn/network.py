#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from .neuron import decision, activate, output
from .backprop import calculate_cursis, calculate_gradient
from .optimizer import sgd, momentum, adagrad, rmsprop, adadelta, adam


class NeuralNetwork(object):
    """
    This class provides essential function for training neural network
     and predicting target with stochastic gradient descent.

    References:
        http://ufldl.stanford.edu/tutorial/
        http://www.slideshare.net/ckmarkohchang/tensorflow-60645128
        https://cs.stanford.edu/~quocle/tutorial1.pdf
        https://cs.stanford.edu/~quocle/tutorial2.pdf
        https://www.cs.cmu.edu/afs/cs/academic/class/15883-f15/slides/backprop.pdf

    Args:
        layer_size (list): Dimension of each layer. It must be
                greater than 2 because you need to provide at least
                dimension of input & output.
        activation (str): determine activate funciton
                sigmoid -> sigmoid (default)
                tanh -> tanh
                relu -> relu 
                other -> no activate
        output (str): determine output funciton
                softmax -> softmax
                sigmoid -> sigmoid (default)
                tanh -> tanh
                relu -> relu
                other -> no output
        loss (str): determine loss funciton
                square -> square_loss (default)
                cross_entropy -> cross_entropy
        solver (str): determine optimization function
                sqd -> stochastit gradient descent(default)
                momentum -> momentum
                adagrad -> adagrad
                rmsprop -> rmsprop
                adadelta -> adadelta
                adam -> adam
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument
        beta (float): decaying average argument
        max_iter (integer): maximum iteration for optimization
        batch_size (integer): mini batch size

    Returns:
        NeuralNetwork object

    Examples:
        >>> nn = NeuralNetwork((2, 2, 2))
        >>> training_X = numpy.array(
                             [[ 1,  1],
                              [-1, -1],
                              [ 1, -1],
                              [-1,  1]]
                         )
        >>> training_Y = numpy.array(
                             [[1],
                              [1],
                              [0],
                              [0]]
                         )
        >>> nn.fit(training_X, training_Y)
        >>> test_X = numpy.array(
                         [[0.9, 0.9]]
                     )
        >>> nn.predict(test_X)
    """
    def __init__(self, layer_sizes, activation='sigmoid',
                 output='sigmoid', loss='square',
                 solver='sgd', learning_rate=1, alpha=0.9,
                 beta=0.999, max_iter=200, batch_size=50):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output = output
        self.loss = loss
        self.solver = solver
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, X, Y):
        """

        Args:
            X (numpy array): M x N, training data
            Y (numpy array): M x 1, target value of training data

        Returns:
            None

        """
        # Initial network
        self._init_model()
        # Init mini batch
        sum_gradients = list()
        for w in self.model['weights']:
            sum_gradients.append(numpy.zeros(w.shape))
        batch_counter = 0
        # Start training
        for it in range(self.max_iter):
            for i in range(X.shape[0]):
                neurons_state = self._transmit(X[i:i+1])
                cursis = calculate_cursis(
                    Y[i:i+1], neurons_state, self.model['weights'],
                    activate_func=self.activation,
                    output_func=self.output,
                    loss_func=self.loss
                )
                gradients = calculate_gradient(neurons_state, cursis)
                sum_gradients, batch_counter = self._mini_batch(
                    gradients, sum_gradients, batch_counter
                )

    def predict(self, X):
        """

        Args:
            X (numpy array): M x N, test data with same dimensions

        Returns:
            predicted (numpy array): M x K, predicted result

        """
        neurons_state = self._transmit(X)
        return neurons_state[-1]['out']

    def _init_model(self):
        """
        Randomly initial weight by given dimension.

        Args:
            layer_size (list): Dimension of each layer. It must be
                    greater than 2 because you need to provide at least
                    dimension of input & output.

        Returns:
            None

        Examples:
        """
        self.model = {
            'weights': list(),
            'factor1': list(),
            'factor2': list()
        }

        for i in range(len(self.layer_sizes) - 1):
            # Initial weight
            temp = numpy.random.randn(
                self.layer_sizes[i] + 1, self.layer_sizes[i + 1]
            )
            self.model['weights'].append(temp)
            # Initial buffer for facto1
            temp = numpy.zeros(
                (self.layer_sizes[i] + 1, self.layer_sizes[i + 1])
            )
            self.model['factor1'].append(temp)
            # Initial buffer for factor2
            if self.solver in ['adadelta', 'adam']:
                temp = numpy.zeros(
                    (self.layer_sizes[i] + 1, self.layer_sizes[i + 1])
                )
                self.model['factor2'].append(temp)
        return self.model

    def _mini_batch(self, gradients, sum_gradients, batch_counter):
        if batch_counter >= self.batch_size:
            for index, item in enumerate(sum_gradients):
                sum_gradients[index] = item / float(self.batch_size)
            self._update_weights(
                sum_gradients, self.solver,
                self.learning_rate, self.alpha, self.beta
            )
            sum_gradients = list()
            for w in self.model['weights']:
                sum_gradients.append(numpy.zeros(w.shape))
            batch_counter = 0
        else:
            for i in range(len(sum_gradients)):
                sum_gradients[i] += gradients[i]
            batch_counter += 1
        return sum_gradients, batch_counter

    def _update_weights(self, gradients, func='sgd',
                        learning_rate=1, alpha=0.9, beta=0.1):
        """

        Args:
            gradients (list):
            learning_rate (float):
            alpha (float): 
            beta (float): 
            func (str): determine activate funciton
                    sgd -> sgd 
                    momentum-> momentum 
                    adagrad -> adagrad
                    rmsprop -> rmsprop
                    adadelta -> adadelta
                    adam -> adam

        Returns:
            self.model
        """
        for i in range(len(self.model['weights'])):
            if func == 'sgd':
                w_update = sgd(gradients[i], learning_rate)
            elif func == 'momentum':
                w_update, f1 = momentum(
                    gradients[i], self.model['factor1'][i],
                    learning_rate, alpha
                )
                self.model['factor1'][i] = f1
            elif func == 'adagrad':
                w_update, f1 = adagrad(
                    gradients[i], self.model['factor1'][i], learning_rate
                )
                self.model['factor1'][i] = f1
            elif func == 'rmsprop':
                w_update, f1 = rmsprop(
                    gradients[i], self.model['factor1'][i],
                    learning_rate, alpha
                )
                self.model['factor1'][i] = f1
            elif func == 'adadelta':
                w_update, f1, f2 = adadelta(
                    gradients[i], self.model['factor1'][i],
                    self.model['factor2'][i], learning_rate, alpha
                )
                self.model['factor1'][i] = f1
                self.model['factor2'][i] = f2
            else:
                w_update, f1, f2 = adam(
                    gradients[i], self.model['factor1'][i],
                    self.model['factor2'][i], learning_rate, alpha, beta
                )
                self.model['factor1'][i] = f1
                self.model['factor2'][i] = f2

            self.model['weights'][i] = self.model['weights'][i] + w_update
        return self.model

    def _transmit(self, x):
        """

        Args:
            x (numpy array): 1 x N, test data

        Returns:
            neurons_state (list): state of each layer

        """
        neurons_state = list()
        # Set x as input neuron
        neurons_state.append({'in': x, 'out': x})
        # Start transmitting
        for i in range(len(self.model['weights']) - 1):
            # Use last neuron's output as input
            z = decision(neurons_state[-1]['out'], self.model['weights'][i])
            # Activate neuron state
            a = activate(z, self.activation)
            neurons_state.append({'in': z, 'out': a})
        # Output layer
        z = decision(neurons_state[-1]['out'], self.model['weights'][-1])
        o = output(z, self.output)
        neurons_state.append({'in': z, 'out': o})
        return neurons_state
