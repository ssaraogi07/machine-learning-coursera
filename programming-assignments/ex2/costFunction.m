function [J, grad] = costFunction(theta, X, Y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(Y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
alpha = 0.2;
main_param = X* theta;
main_param_value = sigmoid(main_param);
second_main_param_value = 1 - main_param_value;
within_brackets = sum(Y.* log(main_param_value) + (1- Y) .* log(second_main_param_value));
J = -1 * within_brackets/m;
subs = main_param_value - Y;

% grad(1) = sum(subs .* X(:,1))/m;
% grad(2) = sum(subs .* X(:,2))/m;
% grad(3) = sum(subs .* X(:,3))/m;

grad = (1/m)*X'*subs;




% =============================================================

end
