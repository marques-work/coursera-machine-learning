function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

    % vector of error coefficients to apply to each element of X
    errors = (X * theta) - y;

    % Represents an n-dimensional vector with respective updates to theta.
    % Essentially, alpha/m * sum(error * x_n) for all n
    upd = (X' * errors) * (alpha / m);

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    % simultaneous update to theta by vector subtraction, i.e.
    % <current theta vector> - <update vector> yields an updated
    % theta vector
    theta = theta - upd;
  end
end
