function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

  % I already vectorized the univariate version, so it doesn't care about the
  % number of features. It should just work.
  J = computeCost(X, y, theta);
end
