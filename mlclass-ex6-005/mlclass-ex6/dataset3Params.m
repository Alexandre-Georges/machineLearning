function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

errors = [];
C_values = [];
sigma_values = [];

possible_values = [0.01 0.03 0.1 0.3 1 3 10 30];

C_index = 1;
while C_index <= size(possible_values, 2),
  sigma_index = 1;

  while sigma_index <= size(possible_values, 2),

    C_values(end + 1) = possible_values(C_index);
    sigma_values(end + 1) = possible_values(sigma_index);

    model = svmTrain(X, y, C_values(end), @(x1, x2) gaussianKernel(x1, x2, sigma_values(end)));
    predictions = svmPredict(model, Xval);
    errors(end + 1) = mean(double(predictions ~= yval));

    sigma_index++;
  end;

  C_index++;
end;

[min_value, min_index] = min(errors);
C = C_values(min_index);
sigma = sigma_values(min_index);

% =========================================================================

end
