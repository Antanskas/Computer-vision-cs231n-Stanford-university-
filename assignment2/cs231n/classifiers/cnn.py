from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        C, H, W = input_dim
        F = num_filters
        H_f = filter_size
        W_f = filter_size

        self.params['W1'] = np.random.randn(F, C, H_f, W_f)*weight_scale
        self.params['b1'] = np.zeros(F)
        self.params['W2'] = np.random.randn(F*(H//2)*(W//2), hidden_dim)*weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale
        self.params['b3'] = np.zeros(num_classes)



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        """
        conv - relu - 2x2 max pool - affine - relu - affine - softmax
        """
        """
        -----------------NAIVE IMPLEMENTATION - SLOW-------------------------

        out, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
        out, cache_relu1 = relu_forward(out)
        out, cache_pool = max_pool_forward_naive(out, pool_param)
        out, cache_affine1 = affine_forward(out, W2, b2)
        out, cache_relu2 = relu_forward(out)
        scores, cache_affine2 = affine_forward(out, W3, b3)
        """

        """
        -----------------FAST IMPLEMENTATION-------------------------
        """
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_pool_out, W2, b2)
        affine_out, affine_cache = affine_forward(affine_relu_out, W3, b3)

        scores = affine_out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        """
        -----------------NAIVE IMPLEMENTATION - SLOW-------------------------

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss  = np.sum((np.sum(np.square(W1)), np.sum(np.square(W2)), np.sum(np.square(W3))))

        dout, dW3, db3 = affine_backward(dscores, cache_affine2)
        dout = relu_backward(dout, cache_relu2)
        dout, dW2, db2 = affine_backward(dout, cache_affine1)
        dout = max_pool_backward_naive(dout, cache_pool)
        dout = relu_backward(dout, cache_relu1)
        dx, dW1, db1 = conv_backward_naive(dout, cache_conv)

        grads['W1'] = dW1 + self.reg*self.params['W1']
        grads['W2'] = dW2 + self.reg*self.params['W2']
        grads['W3'] = dW3 + self.reg*self.params['W3']
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        loss = data_loss + 0.5*self.reg*reg_loss
        """

        """
        -----------------FAST IMPLEMENTATION-------------------------
        """
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss  = np.sum((np.sum(np.square(W1)), np.sum(np.square(W2)), np.sum(np.square(W3))))

        dout, dW3, db3 = affine_backward(dscores, affine_cache)
        dout, dW2, db2 = affine_relu_backward(dout, affine_relu_cache)
        dout, dW1, db1 = conv_relu_pool_backward(dout, conv_relu_pool_cache)

        grads['W1'] = dW1 + self.reg*W1
        grads['W2'] = dW2 + self.reg*W2
        grads['W3'] = dW3 + self.reg*W3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        loss = data_loss + 0.5*self.reg*reg_loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
