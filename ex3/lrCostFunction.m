function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  ytx = y';
  hx = sigmoid(X * theta);

  theta1ToN = [zeros(1, size(theta)(2));theta(2:end, :)];

  J = ((-ytx * log(hx)) - ((1 - ytx) * log(1 - hx)) + sum((lambda / 2) * (theta1ToN .^ 2))) / m;
  grad = ((X' * (hx - y)) + (lambda * theta1ToN)) / m;
end
