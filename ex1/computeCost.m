function J = computeCost(X, y, theta)
  m = length(y); % number of training examples
  squaredErrors = ((X * theta) - y) .^ 2;

  J = sum(squaredErrors) / (2 * m);
end
