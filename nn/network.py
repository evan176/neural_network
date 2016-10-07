#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from .neuron import decision, activate, output
from .backprop import calculate_cursis, calculate_gradient
from .optimizer import sgd, momentum, adagrad, rmsprop, adadelta, adam


class NeuralNetwork(object):
    """This class provides essential function for training neural network
     and predicting target with stochastic gradient descent.

    References:
        http://ufldl.stanford.edu/tutorial/
        http://www.slideshare.net/ckmarkohchang/tensorflow-60645128
        https://cs.stanford.edu/~quocle/tutorial1.pdf
        https://cs.stanford.edu/~quocle/tutorial2.pdf
        https://www.cs.cmu.edu/afs/cs/academic/class/15883-f15/slides/backprop.pdf
        https://github.com/yusugomori/DeepLearning

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
        self._params = {
            'layer_sizes': layer_sizes,
            'activation': activation,
            'output': output,
            'loss': loss,
            'solver': solver,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'beta': beta,
            'max_iter': max_iter,
            'batch_size': batch_size
        }

    @property
    def model(self):
        return self._model['weights']

    def fit(self, X, Y):
        """Fit given data.

        Args:
            X (numpy array): M x N, training data
            Y (numpy array): M x 1, target value of training data

        Returns:
            None

        """
        # Initial network
        self._model = self._init_model()
        # Init mini batch
        self._model = self._train(
            X, Y, self._model, self._params, self._params['max_iter']
        )

    def iterate_fit(self, X, Y, interval):
        """Iteratively fit given data with given interval.

        Args:
            X (numpy array): M x N, training data
            Y (numpy array): M x 1, target value of training data

        Returns:
            weights (list): list of each layer's weight
        """
        # Initial network
        self._model = self._init_model()
        iter_ranges = [interval] * (self._params['max_iter'] / interval)
        iter_ranges.append(self._params['max_iter'] % interval)
        for its in iter_ranges:
            self._model = self._train(
                X, Y, self._model, self._params, its
            )
            yield self.model

    def predict(self, X):
        """

        Args:
            X (numpy array): M x N, test data with same dimensions

        Returns:
            predicted (numpy array): M x K, predicted result

        """
        neurons_state = self._transmit(
            X, self._model['weights'],
            self._params['activation'],
            self._params['output']
        )
        return neurons_state[-1]['out']

    def _init_model(self):
        """
        Randomly initial weight by given dimension.

        Args:
            layer_size (list): Dimension of each layer. It must be
                    greater than 2 because you need to provide at least
                    dimension of input & output.

        Returns:
            model (dict): neural network model
                weights (list): list of each layer's weight
                factor1 (list): buffer for storing meta information
                factor2 (list): buffer for storing meta information
                sum_gradients (list): summarization of gradients
                batch_counter (int): counter for mini batch
        """
        model = {
            'weights': list(),
            'factor1': list(),
            'factor2': list(),
            'sum_gradients': list(),
            'batch_counter': 0
        }

        for i in range(len(self._params['layer_sizes']) - 1):
            # Initial weight
            temp = numpy.random.randn(
                self._params['layer_sizes'][i] + 1,
                self._params['layer_sizes'][i + 1]
            )
            model['weights'].append(temp)
            model['sum_gradients'].append(numpy.zeros(temp.shape))
            # Initial buffer for facto1
            model['factor1'].append(numpy.zeros(temp.shape))
            # Initial buffer for factor2
            if self._params['solver'] in ['adadelta', 'adam']:
                model['factor2'].append(numpy.zeros(temp.shape))
        return model

    def _train(self, X, Y, model, params, iter_num):
        """

        Args:
            X (numpy array):
            Y (label):
            model (dict): neural network model
            params (dict):
            iter_num (int):

        Returns:
            model (dict): neural network model
                weights (list): list of each layer's weight
                factor1 (list): buffer for storing meta information
                factor2 (list): buffer for storing meta information
                sum_gradients (list): summarization of gradients
                batch_counter (int): counter for mini batch
        """
        # Start training
        for it in range(iter_num):
            for i in range(X.shape[0]):
                neurons_state = self._transmit(
                    X[i:i+1], model['weights'],
                    params['activation'],
                    params['output']
                )
                cursis = calculate_cursis(
                    Y[i:i+1], neurons_state, model['weights'],
                    activate_func=params['activation'],
                    output_func=params['output'],
                    loss_func=params['loss']
                )
                gradients = calculate_gradient(neurons_state, cursis)
                model = self._mini_batch(gradients, model, params)
        return model

    def _mini_batch(self, gradients, model, params):
        """

        Args:
            gradients (list): list of each layer's gradient
            model (dict): neural network model
            params (dict):

        Returns:
            model (dict): neural network model
                weights (list): list of each layer's weight
                factor1 (list): buffer for storing meta information
                factor2 (list): buffer for storing meta information
                sum_gradients (list): summarization of gradients
                batch_counter (int): counter for mini batch
        """
        if model['batch_counter'] >= params['batch_size']:
            for index, item in enumerate(model['sum_gradients']):
                model['sum_gradients'][index] = item / float(params['batch_size'])
            model['weights'] = self._update_weights(
                model['sum_gradients'], model['weights'],
                model['factor1'], model['factor2'], params['solver'],
                params['learning_rate'], params['alpha'], params['beta']
            )
            for arr in model['sum_gradients']:
                arr.fill(0.0)
            model['batch_counter'] = 0
        else:
            for i in range(len(gradients)):
                model['sum_gradients'][i] += gradients[i]
            model['batch_counter'] += 1
        return model

    def _update_weights(self, gradients, weights, factor1, factor2,
                        solver='sgd', learning_rate=1, alpha=0.9, beta=0.1):
        """

        Args:
            gradients (list): list of each layer's gradient
            weights (list): list of each layer's weight
            factor1 (list): buffer for storing meta information
            factor2 (list): buffer for storing meta information
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

        Returns:
            weights (list): list of each layer's weight
        """
        for i in range(len(weights)):
            if solver == 'sgd':
                w_update = sgd(gradients[i], learning_rate)
            elif solver == 'momentum':
                w_update, f1 = momentum(
                    gradients[i], factor1[i],
                    learning_rate, alpha
                )
                factor1[i] = f1
            elif solver == 'adagrad':
                w_update, f1 = adagrad(
                    gradients[i], factor1[i], learning_rate
                )
                factor1[i] = f1
            elif solver == 'rmsprop':
                w_update, f1 = rmsprop(
                    gradients[i], factor1[i],
                    learning_rate, alpha
                )
                factor1[i] = f1
            elif solver == 'adadelta':
                w_update, f1, f2 = adadelta(
                    gradients[i], factor1[i],
                    factor2[i], learning_rate, alpha
                )
                factor1[i] = f1
                factor2[i] = f2
            else:
                w_update, f1, f2 = adam(
                    gradients[i], factor1[i],
                    factor2[i], learning_rate, alpha, beta
                )
                factor1[i] = f1
                factor2[i] = f2

            weights[i] = weights[i] + w_update
        return weights

    def _transmit(self, x, weights, activate_func, output_func):
        """

        Args:
            x (numpy array): 1 x N, test data
            weights (list): list of each layer's weight
            activate_func (str): determine activate funciton
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

        Returns:
            neurons_state (list): state of each layer
        """
        neurons_state = list()
        # Set x as input neuron
        neurons_state.append({'in': x, 'out': x})
        # Start transmitting
        for i in range(len(weights) - 1):
            # Use last neuron's output as input
            z = decision(neurons_state[-1]['out'], weights[i])
            # Activate neuron state
            a = activate(z, activate_func)
            neurons_state.append({'in': z, 'out': a})
        # Output layer
        z = decision(neurons_state[-1]['out'], weights[-1])
        o = output(z, output_func)
        neurons_state.append({'in': z, 'out': o})
        return neurons_state
