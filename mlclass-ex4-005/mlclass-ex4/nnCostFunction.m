function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_classes = repmat([1:1:num_labels], m, 1) == repmat(y, 1, num_labels);


X_0 = [ones(m, 1) X];
Hidden_layer = sigmoid(X_0 * Theta1');

Hidden_layer_0 = [ones(m, 1) Hidden_layer];
Output_layer = sigmoid(Hidden_layer_0 * Theta2');

J = 1 / m * sum(sum(-y_classes .* log(Output_layer) - (1 - y_classes) .* log(1 - Output_layer)));

reg = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

J = J + reg;

% -------------------------------------------------------------
Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);

for i = 1:m
  a = [1 X(i, :)];

  Hidden_layer_z = a * Theta1';
  Hidden_layer_a = [1 sigmoid(Hidden_layer_z)];

  Output_layer_z = Hidden_layer_a * Theta2';
  Output_layer_a = sigmoid(Output_layer_z);

  delta_3 = Output_layer_a - y_classes(i, :);
  delta_2 = (delta_3 * Theta2)(2:end) .* sigmoidGradient(Hidden_layer_z);

  Delta_1 = Delta_1 .+ delta_2' * a;
  Delta_2 = Delta_2 .+ delta_3' * Hidden_layer_a;
end;

Theta1_grad = 1 / m * Delta_1;
Theta2_grad = 1 / m * Delta_2;

% =========================================================================

reg_1 = Theta1;
reg_1(:, 1) = zeros(size(reg_1(:, 1)));
reg_1 = reg_1 * lambda / m;

Theta1_grad = Theta1_grad + reg_1;

reg_2 = Theta2;
reg_2(:, 1) = zeros(size(reg_2(:, 1)));
reg_2 = reg_2 * lambda / m;

Theta2_grad = Theta2_grad + reg_2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
