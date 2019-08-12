from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_contributors_count = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # incorrect class gradient part
                dW[:, j] += X[i]
                # count contributor terms to loss function
                loss_contributors_count += 1
        # correct class gradient part
        dW[:, y[i]] += (-1) * loss_contributors_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # num_train = X.shape[0]
    # all_scores = X.dot(W)                                             # dimention N x C
    # true_scores = all_scores[np.arange(num_train), y]                       # dimention N,
    # true_scores = true_scores.reshape(num_train,1)                          # dimention N x 1
    # score_diff = all_scores - true_scores + 1
    # score_diff [np.arange(num_train),y] = 0
    #
    # loss = np.sum(np.maximum(0,score_diff))
    # loss = loss / num_train
    # loss = loss + reg * np.sum(W*W)

    # s: A numpy array of shape (N, C) containing scores
    s = X.dot(W)
    # read correct scores into a column array of height N
    correct_score = s[list(range(num_train)), y]
    correct_score = correct_score.reshape(num_train, -1)
    # subtract correct scores from score matrix and add margin
    s += 1 - correct_score
    # make sure correct scores themselves don't contribute to loss function
    s[list(range(num_train)), y] = 0
    # construct loss function
    loss = np.sum(np.fmax(s, 0)) / num_train
    loss += reg * np.sum(W * W)




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # binary = score_diff
    # binary[binary > 0] = 1
    # row_sum = np.sum(binary, axis=1)
    # binary[np.arange(num_train), y] = -row_sum.T
    # dW = np.dot(X.T, binary)
    # dW /= num_train
    # dW += 2*reg*W

    X_mask = np.zeros(s.shape)
    X_mask[s > 0] = 1
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW