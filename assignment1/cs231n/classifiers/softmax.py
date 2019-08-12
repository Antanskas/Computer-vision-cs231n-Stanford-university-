from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_exam = X.shape[0]
    classes = W.shape[1]
    scores = X.dot(W)
    max_each_example = np.max(scores, axis=1).reshape(num_exam,1)
    scores -= max_each_example

    for i in range(num_exam):
        true_score = np.exp(scores[i,y[i]])
        all_scores = np.sum(np.exp(scores[i,:]))
        softmax_f = true_score/all_scores
        i_exam_loss = -np.log(softmax_f)
        loss += i_exam_loss
        for j in range(classes):
            if j == y[i]:
                dW[:, j] += (softmax_f - 1) * X[i,:]
            else:
                dW[:, j] += (np.exp(scores[i,j])/all_scores) * X[i,:]


    loss /= num_exam
    loss += reg*np.sum(W*W)

    dW /= num_exam
    dW += reg*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_exam = X.shape[0]

    scores = X.dot(W)
    max_each_example = np.max(scores, axis=1).reshape(num_exam,1)
    scores -= max_each_example

    arg1 = np.exp(scores[np.arange(num_exam),y])
    arg2 = np.sum(np.exp(scores), axis=1)
    arg3 = -np.log(arg1/arg2)
    arg4 = np.sum(arg3)/num_exam
    loss = arg4 + 0.5*reg*np.sum(W*W)

    arg11 = np.exp(scores)
    arg22 = np.sum(np.exp(scores), axis=1).reshape((num_exam,1))
    matrix_to_minus_SX = np.zeros_like(scores)
    matrix_to_minus_SX [np.arange(num_exam),y] = 1
    SX = (X.T).dot(arg11/arg22-matrix_to_minus_SX)
    dW = SX
    dW /= num_exam
    dW += reg*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
