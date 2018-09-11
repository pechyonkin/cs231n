import numpy as np
from random import shuffle

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
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # N
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) # vector of C
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] + 1 - correct_class_score # note delta = 1
      if margin > 0:
        loss += margin
        # gradient derived on paper
        dW[:,j]    += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # take average gradient over minibatch
  dW /= num_train
  # add regularization gradient (dL/dW reg * W**2 = 2 * reg * W)
  dW += 2 *reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

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

  num_train = X.shape[0] # N
  train_range = np.arange(num_train)

  # loss = 0.0
  # dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores = np.dot(X, W) #(N, C)
  correct_scores = scores[np.arange(num_train), y].reshape((num_train,1))#(N,1)
  margins = (scores + 1 - correct_scores)
  margins[train_range, y] = 0 # zero out correct margins
  margins = np.maximum(0, margins) # element-wise max(0, ...)

  loss = np.sum(margins) / num_train

  loss += reg * np.sum(W**2)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Before using this approach, a lot of derivation on paper to vectorize

  mask = margins                                            # N x C
  mask[mask > 0] = 1                                        # N x C
  incorrect_margins = np.sum(mask, axis=1)                  # N
  mask[train_range, y] = -1 * incorrect_margins
  dW = np.dot(X.T, mask) / num_train + 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
