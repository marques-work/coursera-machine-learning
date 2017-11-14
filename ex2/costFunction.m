function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  ytx = y';
  hx = sigmoid(X * theta);

  J = (-ytx * log(hx) - ((1 - ytx) * log(1 - hx))) / m;
  grad = (X' * (hx - y)) / m;
end
