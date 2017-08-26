function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.

y = logical(y);
x1 = X(:,1);
x2 = X(:,2);
scatter(x1(y),x2(y),[], '+');
y = ~y;
scatter(x1(y),x2(y),[],'yellow', 'o');







% =========================================================================



hold off;

end
