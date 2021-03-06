import numpy as np
from random import shuffle

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

  num_train = X.shape[0]
  num_class = W.shape[1]

  scores = np.dot(X, W) # N x C

  loss = 0
  dW = np.zeros_like(W)

  for n in range(num_train):
    # Calculating loss
    correct_class = y[n]
    logC = - np.max(scores[n])  # for numerical stability
    total_sum = np.sum(np.exp(scores[n] + logC))
    for c in range(num_class):
      upper_term = np.exp(scores[n,c] + logC)
      # Gradient update
      dW[:, c] += (upper_term / total_sum) * X[n]
      if c == correct_class:
        loss -= np.log((upper_term) / (total_sum))
        # Gradient update: additional term for correct class
        dW[:,c] -= X[n]

  # Average loss and regularization loss
  loss /= num_train
  loss += np.sum(W ** 2)

  # Average gradient and regularization gradient
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  C = W.shape[1]
  D = X.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  logits = np.dot(X, W) # N x C
  logC = np.reshape(-np.max(logits, axis=1), (N, 1)) # N, for numerical stability
  logits += logC
  exp_logits = np.exp(logits)
  probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  losses = -np.log(probs[np.arange(N), y]) # N
  loss = np.sum(losses)

  # Average loss and regularization loss
  loss /= N
  loss += np.sum(W**2)

  # Gradient update
  Y = np.zeros(probs.shape, dtype=y.dtype)
  Y[np.arange(N), y] += 1        # for correct labels (according to derivation)
  dW += np.dot(X.T, (probs - Y)) # DxN.NxC -> DxC
  dW /= N
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

