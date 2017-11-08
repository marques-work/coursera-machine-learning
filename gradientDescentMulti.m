function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

  % I already vectorized the univariate version, so it doesn't care about the
  % number of features. It should just work.
  [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);
end
