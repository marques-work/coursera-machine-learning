function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  [J, grad] = costFunction(theta, X, y);

  theta1toN = [0; theta(2:end)];

  J = J + (lambda * sum(theta1toN .^ 2))/ (2 * m);
  grad = grad + ((lambda * theta1toN) / m);
end
