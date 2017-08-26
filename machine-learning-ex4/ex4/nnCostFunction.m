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
bigDelta_1 = zeros(size(Theta1));
bigDelta_2 = zeros(size(Theta2));
for i = 1:m
    a1 = [1 X(i,:)];
    a1 = a1(:);
    a2 = [1;sigmoid(Theta1 * a1)];
    a3 = sigmoid(Theta2 * a2);
    ly = (1:num_labels) == y(i);
    ly = ly(:);
    delta_3 = a3 - ly;
    delta_2 = ((Theta2(:,(2:end)))' * delta_3) .* sigmoidGradient(Theta1 * a1);
    bigDelta_2 = bigDelta_2 + delta_3 * a2';
    bigDelta_1 = bigDelta_1 + delta_2 * a1'; 
    J = J + (sum((-ly .* log(a3)) - ((1-ly) .* log(1-a3))));
end
J = J/m;
reg_params = [Theta1(:, (2:end)) Theta2(:,(2:end))'];
reg_params = reg_params(:);
J = J + ((lambda/(2*m) * sum(reg_params .^ 2)));

Theta1_grad = ((1/m) * bigDelta_1) + ((lambda/m) * [zeros(size(Theta1,1),1) Theta1(:,(2:end))]);
Theta2_grad = ((1/m) * bigDelta_2) + ((lambda/m) * [zeros(size(Theta2,1),1) Theta2(:,(2:end))]);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
